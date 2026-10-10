import base64

import pytest

from loom.config import Config, EncryptionConfig


@pytest.fixture
def encrypted_config(monkeypatch):
    monkeypatch.setenv("LOOM_TEST_KEY", base64.b64encode(bytes(range(32))).decode())
    return Config(encryption=EncryptionConfig(active_key_id="test", keys={"test": "LOOM_TEST_KEY"}))
