"""Bounded RGB/grayscale PNG grids; importing this module needs no image runtime."""
import json
import math
import struct
import zlib

MAX_COUNT = 64
MAX_ELEMENTS = 4_194_304
MAX_SIDE = 4096
MAX_PIXELS = 4_194_304
MAX_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 65536
SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _chunk(kind, data):
    return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))


def dimensions(width, height, channels):
    if (type(width) is not int or type(height) is not int or type(channels) is not int
            or channels not in (1, 3) or not 1 <= width <= MAX_SIDE
            or not 1 <= height <= MAX_SIDE or width * height > MAX_PIXELS):
        raise ValueError('PNG requires bounded RGB/grayscale dimensions (4096 per side, 4194304 pixels)')


def encode_png(pixels, width, height, channels, metadata=None):
    """Encode tightly packed uint8 rows without external files or executable formats."""
    dimensions(width, height, channels)
    if not isinstance(pixels, bytes) or len(pixels) != width * height * channels:
        raise ValueError('PNG pixel byte count differs from its dimensions')
    meta = json.dumps(metadata or {}, allow_nan=False, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    if len(meta) > MAX_METADATA_BYTES:
        raise ValueError('PNG provenance exceeds the 65536-byte budget')
    stride = width * channels
    rows = b''.join(b'\0' + pixels[start:start + stride] for start in range(0, len(pixels), stride))
    result = (SIGNATURE + _chunk(b'IHDR', struct.pack('!IIBBBBB', width, height, 8, 0 if channels == 1 else 2, 0, 0, 0))
              + _chunk(b'tEXt', b'hypergan\0' + meta)
              + _chunk(b'IDAT', zlib.compress(rows)) + _chunk(b'IEND', b''))
    if len(result) > MAX_BYTES:
        raise ValueError('PNG exceeds the 8388608-byte budget')
    return result


def inspect_png(data):
    """Validate bounded nonanimated 8-bit RGB/grayscale PNG framing and dimensions.

    This deliberately does not decompress image data in the server. Only PNG is
    served inline; HTML/SVG and oversized or malformed PNG headers are rejected.
    """
    if not isinstance(data, bytes) or not 57 <= len(data) <= MAX_BYTES or not data.startswith(SIGNATURE):
        raise ValueError('Invalid bounded PNG artifact')
    offset, header, image, ended = 8, None, False, False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError('Truncated PNG chunk')
        size = struct.unpack('!I', data[offset:offset + 4])[0]
        end = offset + size + 12
        if end > len(data):
            raise ValueError('Truncated PNG chunk')
        kind, payload = data[offset + 4:offset + 8], data[offset + 8:end - 4]
        if zlib.crc32(kind + payload) != struct.unpack('!I', data[end - 4:end])[0]:
            raise ValueError('PNG chunk checksum mismatch')
        if header is None:
            if kind != b'IHDR' or size != 13:
                raise ValueError('PNG requires a first IHDR chunk')
            width, height, depth, color, compression, filtering, interlace = struct.unpack('!IIBBBBB', payload)
            channels = {0: 1, 2: 3}.get(color, 0)
            dimensions(width, height, channels)
            if (depth, compression, filtering, interlace) != (8, 0, 0, 0):
                raise ValueError('PNG must be noninterlaced 8-bit RGB or grayscale')
            header = {'width': width, 'height': height, 'channels': channels}
        elif kind == b'IDAT':
            image = True
        elif kind == b'IEND':
            if size or not image or end != len(data):
                raise ValueError('Invalid PNG ending')
            ended = True
        elif kind != b'tEXt' or size > MAX_METADATA_BYTES + 9:
            raise ValueError('Unsupported PNG chunk')
        offset = end
    if not ended:
        raise ValueError('PNG is missing its ending')
    return header


def tensor_grid(values, metadata=None):
    """Clamp [-1,1], round (x+1)*127.5, and tile images in row-major order.

    Only copied CPU values are transformed. No RNG, model or input tensor changes.
    Custom generator allocations before this function remain caller-owned.
    """
    import torch
    if (not isinstance(values, torch.Tensor) or values.ndim != 4 or values.shape[1] not in (1, 3)
            or not values.is_floating_point()):
        raise ValueError('PNG sampling requires floating NCHW RGB/grayscale images in [-1,1]')
    count, channels, height, width = values.shape
    if not 1 <= count <= MAX_COUNT or values.numel() > MAX_ELEMENTS:
        raise ValueError('PNG sampling exceeds 64 images or 4194304 tensor elements')
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    dimensions(columns * width, rows * height, channels)
    if not torch.isfinite(values).all():
        raise ValueError('PNG contains nonfinite values')
    pixels = values.detach().to(device='cpu', dtype=torch.float32).clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
    grid = torch.zeros((rows * height, columns * width, channels), dtype=torch.uint8)
    for index, value in enumerate(pixels):
        row, column = divmod(index, columns)
        grid[row * height:(row + 1) * height, column * width:(column + 1) * width] = value.permute(1, 2, 0)
    descriptor = {'width': columns * width, 'height': rows * height, 'channels': channels,
                  'rows': rows, 'columns': columns, 'count': count,
                  'pixel_conversion': 'clamp[-1,1]; round((x+1)*127.5)', 'empty_cells': 'black'}
    encoded = encode_png(grid.numpy().tobytes(), descriptor['width'], descriptor['height'], channels,
                         dict(metadata or {}, grid=descriptor))
    return encoded, descriptor
