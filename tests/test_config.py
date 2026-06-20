import importlib


def test_defaults_when_env_absent(monkeypatch):
    for k in ("CF_HANDLE", "DATABASE_URL", "TARGET_RATING", "TARGET_DATE", "WEEKLY_HOURS"):
        monkeypatch.delenv(k, raising=False)
    import config
    importlib.reload(config)
    s = config.Settings(_env_file=None)
    assert s.target_rating == 1900
    assert s.weekly_hours == 8
    assert s.target_date == "2026-12-21"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CF_HANDLE", "tourist")
    monkeypatch.setenv("TARGET_RATING", "2400")
    import config
    importlib.reload(config)
    s = config.Settings(_env_file=None)
    assert s.cf_handle == "tourist"
    assert s.target_rating == 2400


def test_pedagogical_constants_present():
    import config
    importlib.reload(config)
    assert config.MAX_CONSECUTIVE_SAME_TAG == 1
    assert config.TRAIN_TARGET_BAND == (0.55, 0.80)
