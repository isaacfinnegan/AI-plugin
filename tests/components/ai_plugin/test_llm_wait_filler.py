"""Tests for the opt-in transient LLM wait filler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.ai_plugin.const import CONF_LLM_WAIT_FILLER
from custom_components.ai_plugin.orchestrator import core
from custom_components.ai_plugin.exceptions import OrchestratorError

from .test_conversation import _make_mock_entry


class _VoiceOrchestrator:
    """Minimal orchestrator that reaches the LLM-start hook once."""

    def __init__(self, *, raises: bool = False) -> None:
        self.is_voice_device = MagicMock(return_value=True)
        self._raises = raises
        self.async_process = AsyncMock(side_effect=self._process)

    async def _process(self, **kwargs):
        callback = kwargs.get("on_llm_start")
        if callback is not None:
            callback()
        if self._raises:
            raise OrchestratorError("provider unavailable")
        return "The answer."


def _entity(hass, *, enabled: bool, orchestrator):
    from custom_components.ai_plugin.conversation import AIPluginConversationEntity

    entity = AIPluginConversationEntity.__new__(AIPluginConversationEntity)
    entity.hass = hass
    entity._entry = _make_mock_entry({CONF_LLM_WAIT_FILLER: enabled})
    entity._attr_unique_id = "test"
    entity._orchestrator = orchestrator
    entity._cancel_follow_up = MagicMock()
    entity._satellite_history_id = lambda device_id, conversation_id, language: conversation_id
    entity._continued_conversation_id = lambda device_id, conversation_id: conversation_id
    entity._count_chain_turn = MagicMock(return_value=0)
    entity._recent_replies = {}
    entity._remember_reply = MagicMock(wraps=entity._remember_reply)
    entity._reset_chain_turns = MagicMock()
    entity._follow_up_conv_store = lambda: {}
    entity._enrollment_active = MagicMock(return_value=False)
    return entity


def _input():
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context

    return ConversationInput(
        text="Tell me something",
        context=Context(),
        conversation_id="filler-conversation",
        device_id="satellite-device",
        language="en",
        agent_id=None,
    )


async def test_filler_is_one_transient_event_not_history_or_model_input(hass) -> None:
    from homeassistant.components.conversation import ChatLog

    orch = _VoiceOrchestrator()
    entity = _entity(hass, enabled=True, orchestrator=orch)
    chat_log = ChatLog("filler-conversation")
    events = []
    chat_log.delta_listener = lambda log, delta: events.append((log, delta))

    result = await entity._async_handle_message(_input(), chat_log)

    assert result.response.speech["plain"]["speech"] == "The answer."
    # HA starts a single streaming-TTS input after the filler. A buffered
    # OpenAI-compatible backend never invokes on_delta, so the final must be
    # explicitly sent to that already-open stream exactly once.
    assert [event[1] for event in events] == [
        {
            "role": "assistant",
            "content": "One moment while I check that. ".ljust(61),
        },
    ]
    filler = events[0][1]
    assert filler["role"] == "assistant"
    assert filler["content"].startswith("One moment while I check that.")
    assert len(filler["content"]) == 61
    assert chat_log.content == [{"role": "assistant", "content": "The answer."}]
    assert chat_log.deltas == ["The answer."]
    # Although transient to ChatLog/model history, the audible filler is an
    # echo candidate so a satellite cannot execute its own preamble.
    entity._remember_reply.assert_any_call("satellite-device", filler["content"])
    assert entity._recent_replies_for("satellite-device") == [
        filler["content"], "The answer."
    ]
    # This is enough text for the normal self-echo matcher, so a re-heard
    # filler is dropped before it can become a command.
    from custom_components.ai_plugin.conversation import _is_self_echo
    assert _is_self_echo(filler["content"], entity._recent_replies_for("satellite-device"))
    kwargs = orch.async_process.await_args.kwargs
    assert kwargs["on_delta"] is not None
    assert kwargs["on_llm_start"] is not None


async def test_pipeline_listener_gets_filler_then_final_stream_ends(hass) -> None:
    """Mirror HA's pipeline wiring: listener is distinct from ChatLog history.

    The listener receives the transient preamble first. The normal delta stream
    contains only the final reply and must receive its terminating sentinel so
    the pipeline does not wait indefinitely for more audio.
    """
    from homeassistant.components.conversation import ChatLog

    class PipelineChatLog(ChatLog):
        def __init__(self):
            super().__init__("filler-conversation")
            self.pipeline_events: list[dict] = []
            self.stream_finished = False
            self.delta_listener = lambda _log, delta: self.pipeline_events.append(delta)

        async def async_add_delta_content_stream(self, agent_id, stream):
            text = ""
            async for delta in stream:
                # HA's stream path invokes the same pipeline listener for
                # ordinary final-reply deltas. The direct filler event must
                # therefore precede the final reply event.
                self.delta_listener(self, delta)
                text += delta.get("content", "")
            self.stream_finished = True
            self.content.append({"role": "assistant", "content": text})
            if False:
                yield None

    async def process(**kwargs):
        kwargs["on_llm_start"]()
        kwargs["on_delta"]("The final answer.")
        return "The final answer."

    orch = _VoiceOrchestrator()
    orch.async_process = AsyncMock(side_effect=process)
    entity = _entity(hass, enabled=True, orchestrator=orch)
    chat_log = PipelineChatLog()

    result = await entity._async_handle_message(_input(), chat_log)

    assert result.response.speech["plain"]["speech"] == "The final answer."
    assert [
        event["content"] for event in chat_log.pipeline_events if "content" in event
    ] == [
        "One moment while I check that. ".ljust(61),
        "The final answer.",
    ]
    assert chat_log.stream_finished is True
    assert chat_log.content == [{"role": "assistant", "content": "The final answer."}]
    assert entity._recent_replies_for("satellite-device") == [
        "One moment while I check that. ".ljust(61), "The final answer."
    ]


async def test_filler_is_disabled_by_default(hass) -> None:
    from homeassistant.components.conversation import ChatLog

    orch = _VoiceOrchestrator()
    entity = _entity(hass, enabled=False, orchestrator=orch)
    chat_log = ChatLog("filler-conversation")
    events = []
    chat_log.delta_listener = lambda log, delta: events.append(delta)

    await entity._async_handle_message(_input(), chat_log)

    assert events == []
    assert orch.async_process.await_args.kwargs["on_llm_start"] is None


async def test_local_shortcut_does_not_emit_filler(hass) -> None:
    from homeassistant.components.conversation import ChatLog

    orch = _VoiceOrchestrator()
    entity = _entity(hass, enabled=True, orchestrator=orch)
    chat_log = ChatLog("filler-conversation")
    events = []
    chat_log.delta_listener = lambda log, delta: events.append(delta)

    # The hook is only called at the LLM boundary; a local shortcut returns
    # before it, exactly as the real orchestrator's pre-LLM shortcuts do.
    async def local_shortcut(**kwargs):
        assert kwargs["on_llm_start"] is not None
        return "Shortcut answer."

    orch.async_process = AsyncMock(side_effect=local_shortcut)
    result = await entity._async_handle_message(_input(), chat_log)

    assert result.response.speech["plain"]["speech"] == "Shortcut answer."
    assert events == []
    assert chat_log.content == []


async def test_buffered_provider_error_reaches_the_same_tts_stream(hass) -> None:
    """A filler-started HA stream must receive the surfaced provider error too."""
    from homeassistant.components.conversation import ChatLog

    orch = _VoiceOrchestrator(raises=True)
    entity = _entity(hass, enabled=True, orchestrator=orch)

    class PipelineChatLog(ChatLog):
        def __init__(self):
            super().__init__("filler-conversation")
            self.pipeline_events: list[dict] = []
            self.delta_listener = lambda _log, delta: self.pipeline_events.append(delta)

        async def async_add_delta_content_stream(self, agent_id, stream):
            text = ""
            async for delta in stream:
                self.delta_listener(self, delta)
                text += delta.get("content", "")
            self.content.append({"role": "assistant", "content": text})
            if False:
                yield None

    chat_log = PipelineChatLog()
    result = await entity._async_handle_message(_input(), chat_log)

    error = result.response.speech["plain"]["speech"]
    assert "provider unavailable" in error
    assert [event.get("content") for event in chat_log.pipeline_events if "content" in event] == [
        "One moment while I check that. ".ljust(61),
        error,
    ]
    assert chat_log.content == [{"role": "assistant", "content": error}]


async def test_streaming_provider_final_is_not_duplicated_after_filler(hass) -> None:
    """A real provider delta already carries the final; do not append it again."""
    from homeassistant.components.conversation import ChatLog

    async def process(**kwargs):
        kwargs["on_llm_start"]()
        kwargs["on_delta"]("The streamed answer.")
        return "The streamed answer."

    orch = _VoiceOrchestrator()
    orch.async_process = AsyncMock(side_effect=process)
    entity = _entity(hass, enabled=True, orchestrator=orch)
    chat_log = ChatLog("filler-conversation")
    events = []
    chat_log.delta_listener = lambda _log, delta: events.append(delta)

    result = await entity._async_handle_message(_input(), chat_log)

    assert result.response.speech["plain"]["speech"] == "The streamed answer."
    assert chat_log.deltas == ["The streamed answer."]
    assert chat_log.content == [{"role": "assistant", "content": "The streamed answer."}]
    assert [event["content"] for event in events] == [
        "One moment while I check that. ".ljust(61),
    ]


async def test_filler_listener_failures_do_not_hide_llm_errors(hass) -> None:
    from homeassistant.components.conversation import ChatLog

    orch = _VoiceOrchestrator(raises=True)
    entity = _entity(hass, enabled=True, orchestrator=orch)
    chat_log = ChatLog("filler-conversation")

    def broken_listener(log, delta):
        raise RuntimeError("stream unavailable")

    chat_log.delta_listener = broken_listener
    result = await entity._async_handle_message(_input(), chat_log)

    speech = result.response.speech["plain"]["speech"]
    assert "provider unavailable" in speech
    # The final error is still persisted through the ordinary delta stream;
    # the failed direct filler listener never pollutes it.
    assert chat_log.content == [{"role": "assistant", "content": speech}]
    assert chat_log.deltas == [speech]


async def test_non_voice_request_never_receives_filler(hass) -> None:
    from homeassistant.components.conversation import ChatLog

    orch = _VoiceOrchestrator()
    orch.is_voice_device.return_value = False
    entity = _entity(hass, enabled=True, orchestrator=orch)
    chat_log = ChatLog("filler-conversation")
    events = []
    chat_log.delta_listener = lambda log, delta: events.append(delta)

    await entity._async_handle_message(_input(), chat_log)

    assert events == []
    assert orch.async_process.await_args.kwargs["on_llm_start"] is None


async def test_orchestrator_calls_hook_only_after_local_shortcuts_miss() -> None:
    from .test_conversation import _baseline_orch

    orch = _baseline_orch()
    orch._hass = MagicMock()
    orch._provider = MagicMock()
    orch._provider.async_complete = AsyncMock(return_value="LLM answer.")
    started = MagicMock()

    with patch.object(core, "try_shortcut", return_value="Local answer."):
        reply = await orch.async_process("local question", "local", "en", on_llm_start=started)

    assert reply == "Local answer."
    started.assert_not_called()
    orch._provider.async_complete.assert_not_awaited()


async def test_orchestrator_hook_failure_does_not_block_llm() -> None:
    from .test_conversation import _baseline_orch

    orch = _baseline_orch()
    orch._provider = MagicMock()
    orch._provider.async_complete = AsyncMock(return_value="LLM answer.")

    def broken_hook():
        raise RuntimeError("progress failed")

    reply = await orch.async_process("ordinary question", "llm", "en", on_llm_start=broken_hook)

    assert reply == "LLM answer."
    orch._provider.async_complete.assert_awaited_once()
