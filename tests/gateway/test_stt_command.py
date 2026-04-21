"""Tests for gateway /stt command (toggle stt.send_transcription via /stt echo, persist to config.yaml)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/stt", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(config: GatewayConfig | None = None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = config if config is not None else GatewayConfig()
    runner.adapters = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    return runner


class TestStatusSubcommand:

    @pytest.mark.asyncio
    async def test_bare_command_shows_status(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner(GatewayConfig(
            stt_enabled=True,
            stt_send_transcription=True,
            stt_send_transcription_header="🎤 Heard:\n\n",
        ))

        result = await runner._handle_stt_command(_make_event("/stt"))

        assert "STT Status" in result
        assert "Auto-transcribe inbound voice" in result
        assert "Echo transcription back to chat" in result
        assert "ON" in result

    @pytest.mark.asyncio
    async def test_status_subcommand_equivalent_to_bare(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()
        result = await runner._handle_stt_command(_make_event("/stt status"))
        assert "STT Status" in result

    @pytest.mark.asyncio
    async def test_status_shows_none_for_empty_header(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner(GatewayConfig(stt_send_transcription_header=""))
        result = await runner._handle_stt_command(_make_event("/stt"))
        assert "(none)" in result


class TestEchoToggle:

    @pytest.mark.asyncio
    async def test_toggle_from_off_to_on(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("stt:\n  enabled: true\n", encoding="utf-8")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner(GatewayConfig(stt_send_transcription=False))
        result = await runner._handle_stt_command(_make_event("/stt echo"))

        assert "ON" in result
        assert runner.config.stt_send_transcription is True
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["stt"]["send_transcription"] is True

    @pytest.mark.asyncio
    async def test_toggle_from_on_to_off(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "stt:\n  enabled: true\n  send_transcription: true\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner(GatewayConfig(stt_send_transcription=True))
        result = await runner._handle_stt_command(_make_event("/stt echo"))

        assert "OFF" in result
        assert runner.config.stt_send_transcription is False
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["stt"]["send_transcription"] is False

    @pytest.mark.asyncio
    async def test_explicit_on(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner(GatewayConfig(stt_send_transcription=True))
        # Already on — explicit "on" should keep it on (idempotent)
        result = await runner._handle_stt_command(_make_event("/stt echo on"))

        assert "ON" in result
        assert runner.config.stt_send_transcription is True
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["stt"]["send_transcription"] is True

    @pytest.mark.asyncio
    async def test_explicit_off(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner(GatewayConfig(stt_send_transcription=False))
        result = await runner._handle_stt_command(_make_event("/stt echo off"))

        assert "OFF" in result
        assert runner.config.stt_send_transcription is False
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["stt"]["send_transcription"] is False

    @pytest.mark.asyncio
    async def test_echo_status_subcommand(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner(GatewayConfig(stt_send_transcription=True))
        result = await runner._handle_stt_command(
            _make_event("/stt echo status")
        )
        assert "ON" in result
        # status subcommand should NOT mutate state
        assert runner.config.stt_send_transcription is True

    @pytest.mark.asyncio
    async def test_repeated_toggle_flips_back_and_forth(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("stt:\n  enabled: true\n", encoding="utf-8")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner(GatewayConfig(stt_send_transcription=False))

        expected = [True, False, True, False]
        for want in expected:
            await runner._handle_stt_command(_make_event("/stt echo"))
            saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            assert saved["stt"]["send_transcription"] is want
            assert runner.config.stt_send_transcription is want

    @pytest.mark.asyncio
    async def test_preserves_other_stt_keys(self, tmp_path, monkeypatch):
        """Toggling send_transcription must not clobber sibling stt.* keys."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "stt:\n"
            "  enabled: true\n"
            "  provider: openai\n"
            "  send_transcription_header: \"🎤 Heard:\\n\\n\"\n"
            "  openai:\n"
            "    model: whisper-1\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner(GatewayConfig(stt_send_transcription=False))
        await runner._handle_stt_command(_make_event("/stt echo on"))

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["stt"]["send_transcription"] is True
        assert saved["stt"]["enabled"] is True
        assert saved["stt"]["provider"] == "openai"
        assert saved["stt"]["send_transcription_header"] == "🎤 Heard:\n\n"
        assert saved["stt"]["openai"] == {"model": "whisper-1"}

class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_unknown_subcommand_returns_usage(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()
        result = await runner._handle_stt_command(_make_event("/stt bogus"))
        assert "Unknown subcommand" in result
        assert "echo" in result

    @pytest.mark.asyncio
    async def test_unknown_action_returns_usage(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        runner = _make_runner()
        result = await runner._handle_stt_command(
            _make_event("/stt echo maybe")
        )
        assert "Unknown option" in result

    @pytest.mark.asyncio
    async def test_creates_stt_block_if_missing(self, tmp_path, monkeypatch):
        """If config.yaml has no stt: block, the command creates it."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("voice:\n  record_key: ctrl+b\n", encoding="utf-8")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner(GatewayConfig(stt_send_transcription=False))
        await runner._handle_stt_command(_make_event("/stt echo on"))

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["stt"] == {"send_transcription": True}
        assert saved["voice"] == {"record_key": "ctrl+b"}  # sibling preserved


def test_stt_is_in_gateway_known_commands():
    """The /stt command is recognized by the gateway dispatch."""
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
    assert "stt" in GATEWAY_KNOWN_COMMANDS


def test_stt_has_echo_subcommand():
    """Tab-complete should know about 'echo'."""
    from hermes_cli.commands import SUBCOMMANDS
    assert "echo" in SUBCOMMANDS.get("/stt", [])
