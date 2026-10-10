from __future__ import annotations

from loom.contracts import (
    canonical,
)
from loom.security.crypto import ResourceType
from loom.storage.base import Storage
from loom.storage.checkpoint import (
    Image,
)


class JSONStorage(Storage):
    kind: ResourceType = "storage_json"
    maximum = 16 * 1024**2

    def decode(self, data: bytes) -> Image:
        return Image.model_validate_json(data)

    def encode(self, image: Image) -> bytes:
        return canonical(image, max_bytes=self.maximum)
