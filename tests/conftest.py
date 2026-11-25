import pytest

from idea_miner import config


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Every test gets its own empty database and specs folder."""
    data = tmp_path / "data"
    specs = tmp_path / "specs"
    data.mkdir()
    specs.mkdir()
    monkeypatch.setattr(config, "data_dir", lambda: data)
    monkeypatch.setattr(config, "specs_dir", lambda: specs)
    monkeypatch.setenv("IDEA_MINER_SALT", "test-salt")
    yield
