"""Gateway STT config tests — honor stt.enabled: false from config.yaml."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def test_gateway_config_stt_disabled_from_dict_nested():
    config = GatewayConfig.from_dict({"stt": {"enabled": False}})
    assert config.stt_enabled is False


def test_gateway_config_stt_send_transcription_defaults_and_nested_yaml():
    """Defaults are off + empty header; both keys load from the nested stt.* block."""
    default = GatewayConfig.from_dict({})
    assert default.stt_send_transcription is False
    assert default.stt_send_transcription_header == ""

    configured = GatewayConfig.from_dict(
        {"stt": {"send_transcription": True, "send_transcription_header": "🎤 Heard:\n\n"}}
    )
    assert configured.stt_send_transcription is True
    assert configured.stt_send_transcription_header == "🎤 Heard:\n\n"


def test_load_gateway_config_bridges_stt_enabled_from_config_yaml(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.dump({"stt": {"enabled": False}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    config = load_gateway_config()

    assert config.stt_enabled is False


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_skips_when_stt_disabled():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=False)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("transcribe_audio should not be called when STT is disabled"),
    ):
        result = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "transcription is disabled" in result.lower()
    assert "caption" in result


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_avoids_bogus_no_provider_message_for_backend_key_errors():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": False, "error": "VOICE_TOOLS_OPENAI_KEY not set"},
    ):
        result = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "No STT provider is configured" not in result
    assert "trouble transcribing" in result
    assert "caption" in result


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_transcribes_queued_voice_event():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/queued-voice.ogg"],
        media_types=["audio/ogg"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "queued voice transcript",
            "provider": "local_command",
        },
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "queued voice transcript" in result
    assert "voice message" in result.lower()


# --- Integration tests for stt_send_transcription (echo) ---


def _echo_runner(*, enabled: bool, header: str = "", send_side_effect=None):
    """Build a GatewayRunner wired to a mock Telegram adapter for echo tests."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        stt_enabled=True,
        stt_send_transcription=enabled,
        stt_send_transcription_header=header,
    )
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123456", chat_type="dm")
    adapter = AsyncMock()
    if send_side_effect is not None:
        adapter.send.side_effect = send_side_effect
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner, source, adapter


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_echoes_with_configured_header():
    """When enabled, the transcript is echoed via the adapter with the configured header prepended."""
    runner, source, adapter = _echo_runner(
        enabled=True, header="🎤 **Voice Transcription**\n\n"
    )
    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello world"},
    ):
        result = await runner._enrich_message_with_transcription(
            "", ["/tmp/voice.ogg"], source=source,
        )

    assert "hello world" in result
    adapter.send.assert_called_once()
    assert adapter.send.call_args.args == (
        "123456",
        "🎤 **Voice Transcription**\n\nhello world",
    )


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_does_not_echo_when_disabled():
    runner, source, adapter = _echo_runner(enabled=False)
    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello world"},
    ):
        result = await runner._enrich_message_with_transcription(
            "", ["/tmp/voice.ogg"], source=source,
        )

    assert "hello world" in result
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_tolerates_echo_send_failure():
    """A failing echo send must not break the enriched transcript returned to the agent."""
    runner, source, _adapter = _echo_runner(
        enabled=True, send_side_effect=Exception("network error"),
    )
    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello world"},
    ):
        result = await runner._enrich_message_with_transcription(
            "", ["/tmp/voice.ogg"], source=source,
        )

    assert "hello world" in result
