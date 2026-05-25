from __future__ import annotations

from src.utils import env_config


def test_env_local_overrides_env_file_without_process_override(tmp_path, monkeypatch) -> None:
    key = "CODEX_TEST_ENV_PRIORITY"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(f"{key}=from_env\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text(f"{key}=from_local\n", encoding="utf-8")

    env_config._ENV_LOADED = False

    assert env_config.get_env(key) == "from_local"


def test_process_env_overrides_env_local(tmp_path, monkeypatch) -> None:
    key = "CODEX_TEST_PROCESS_ENV_PRIORITY"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(key, "from_process")
    (tmp_path / ".env").write_text(f"{key}=from_env\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text(f"{key}=from_local\n", encoding="utf-8")

    env_config._ENV_LOADED = False

    assert env_config.get_env(key) == "from_process"
