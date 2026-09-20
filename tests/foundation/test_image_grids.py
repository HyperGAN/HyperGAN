"""PNG framing and web safety do not need Torch or Pillow."""
import json
import struct
import zlib

import pytest

from hypergan.image_grids import _chunk, encode_png, inspect_png, SIGNATURE


def test_png_has_exact_pixels_and_embedded_provenance():
    pixels = bytes([0, 127, 255, 255, 128, 0])
    encoded = encode_png(pixels, 2, 1, 3, {'step': 8, 'identity': {'attempt_id': 'α'}})
    assert inspect_png(encoded) == {'width': 2, 'height': 1, 'channels': 3}
    offset, chunks = 8, {}
    while offset < len(encoded):
        size = struct.unpack('!I', encoded[offset:offset + 4])[0]
        chunks[encoded[offset + 4:offset + 8]] = encoded[offset + 8:offset + 8 + size]
        offset += size + 12
    assert zlib.decompress(chunks[b'IDAT']) == b'\0' + pixels
    assert json.loads(chunks[b'tEXt'].split(b'\0', 1)[1]) == {'step': 8, 'identity': {'attempt_id': 'α'}}


@pytest.mark.parametrize('data', [b'<svg onload="alert(1)"></svg>', b'not png',
    SIGNATURE + _chunk(b'IHDR', struct.pack('!IIBBBBB', 100000, 100000, 8, 2, 0, 0, 0))
    + _chunk(b'IDAT', b'x') + _chunk(b'IEND', b''),
    encode_png(b'\0', 1, 1, 1)[:-1], encode_png(b'\0', 1, 1, 1) + b'extra'])
def test_untrusted_image_artifact_rejects_non_png_oversize_and_corruption(data):
    with pytest.raises(ValueError):
        inspect_png(data)


def test_encoder_limits_before_allocating_rows():
    with pytest.raises(ValueError, match='dimensions'):
        encode_png(b'', 4097, 1, 3)
    with pytest.raises(ValueError, match='pixel byte count'):
        encode_png(b'', 1, 1, 3)
    with pytest.raises(ValueError, match='provenance'):
        encode_png(b'\0', 1, 1, 1, {'text': 'x' * 65536})
