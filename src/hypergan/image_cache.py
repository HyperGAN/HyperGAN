"""Disposable, content-checked uint8 pixels; never a replacement for sources."""
import hashlib
import logging
import os
import tempfile
from pathlib import Path

_MAGIC = b'HGPIX01\0'


class PixelCache:
    """Atomic per-image files shared by decoder threads and independent loaders.

    The namespace pins preprocessing and Pillow version. The source digest pins
    the input. A checksum binds both to the output bytes, detecting partial or
    corrupted cache files. A miss always falls back to the original decoder.
    """

    def __init__(self, directory, namespace, size):
        self.root = Path(directory).expanduser().resolve() / namespace
        self.namespace, self.size = namespace, size
        self._write_enabled = True

    def _path(self, source_sha256):
        return self.root / source_sha256[:2] / (source_sha256[2:] + '.pixels')

    def _checksum(self, source_sha256, pixels):
        return hashlib.sha256((self.namespace + source_sha256).encode('ascii') + pixels).digest()

    def get(self, source_sha256):
        try:
            with self._path(source_sha256).open('rb') as stream:
                content = stream.read(self.size + 41)
        except OSError:
            return None
        if len(content) != self.size + 40 or content[:8] != _MAGIC:
            return None
        pixels = content[40:]
        return pixels if content[8:40] == self._checksum(source_sha256, pixels) else None

    def put(self, source_sha256, pixels):
        if not self._write_enabled:
            return
        if len(pixels) != self.size:
            raise ValueError('Pixel cache received an unexpected image size')
        temporary = None
        try:
            path = self._path(source_sha256)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.pixels-', dir=path.parent)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(_MAGIC + self._checksum(source_sha256, pixels) + pixels)
            # Concurrent writers produce identical complete bytes. A process
            # crash cannot expose a partial published entry; no fsync is needed
            # for disposable data that is checked and regenerated on demand.
            os.replace(temporary, path)
        except OSError as exc:
            self._write_enabled = False
            logging.getLogger(__name__).warning(
                'Pixel cache writes disabled for %s: %s; using source decoding', self.root, exc)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
