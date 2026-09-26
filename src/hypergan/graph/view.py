"""Viewable values that samplers return, and how they are written to files.

The core never assumes images. A sampler maps arbitrary model outputs to one of
these (or a dict of them); each has a plain file form the viewer can show:
image grid (PNG), 2-D points (SVG + JSON), audio (WAV), text (TXT) or a
generic tensor (JSON).
"""
import json
import math
from pathlib import Path


def image(values, low=-1.0, high=1.0):
    """NCHW images (1 or 3 channels) in ``[low, high]``."""
    return {"kind": "image", "value": values, "low": low, "high": high}


def points(values):
    """An (N, 2) tensor of points, e.g. samples of a 2-D distribution."""
    return {"kind": "points", "value": values}


def audio(values, rate):
    """(N, T) or (N, C, T) waveforms in [-1, 1] at ``rate`` Hz."""
    return {"kind": "audio", "value": values, "rate": int(rate)}


def text(values):
    """A string or list of strings."""
    return {"kind": "text", "value": values}


def tensor(values):
    """Any tensor or nested dict/list of tensors, written as JSON."""
    return {"kind": "tensor", "value": values}


def infer(value):
    """Wrap a raw sampler result: NCHW 1/3-channel -> image, (N, 2) -> points, else tensor."""
    import torch
    if isinstance(value, dict) and "kind" in value and "value" in value:
        return value
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        if value.ndim == 4 and value.shape[1] in (1, 3):
            return image(value)
        if value.ndim == 2 and value.shape[1] == 2:
            return points(value)
    return tensor(value)


def _plain(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _svg(xy, size=320):
    finite = [(x, y) for x, y in xy if math.isfinite(x) and math.isfinite(y)]
    if not finite:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"/>'
    xs, ys = [p[0] for p in finite], [p[1] for p in finite]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    scale = (size - 20) / span
    dots = "".join(f'<circle cx="{size / 2 + (x - cx) * scale:.1f}" cy="{size / 2 - (y - cy) * scale:.1f}" r="2"/>'
                   for x, y in finite)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<rect width="100%" height="100%" fill="white"/><g fill="#3a5ba0" fill-opacity="0.7">{dots}</g></svg>')


def write(directory, name, view, metadata=None):
    """Write one viewable; returns the paths written."""
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    kind, value = view["kind"], view["value"]
    written = []
    if kind == "image":
        from ..image_grids import tensor_grid, MAX_COUNT
        low, high = view["low"], view["high"]
        scaled = (value.detach().float().cpu()[:MAX_COUNT] - low) / (high - low) * 2 - 1
        encoded, _ = tensor_grid(scaled, dict(metadata or {}, name=name))
        path = directory / f"{name}.png"
        path.write_bytes(encoded)
        written.append(path)
    elif kind == "points":
        xy = value.detach().float().cpu().tolist()
        (directory / f"{name}.svg").write_text(_svg(xy))
        (directory / f"{name}.json").write_text(json.dumps({"kind": "points", "points": xy, **(metadata or {})}))
        written += [directory / f"{name}.svg", directory / f"{name}.json"]
    elif kind == "audio":
        import wave
        waves = value.detach().float().cpu()
        if waves.ndim == 3:
            waves = waves.mean(1)
        for index, wave_value in enumerate(waves):
            path = directory / f"{name}-{index:03d}.wav"
            pcm = (wave_value.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(view["rate"])
                stream.writeframes(pcm)
            written.append(path)
    elif kind == "text":
        lines = value if isinstance(value, (list, tuple)) else [value]
        path = directory / f"{name}.txt"
        path.write_text("\n".join(str(line) for line in lines) + "\n")
        written.append(path)
    elif kind == "tensor":
        path = directory / f"{name}.json"
        path.write_text(json.dumps({"kind": "tensor", "value": _plain(value), **(metadata or {})}))
        written.append(path)
    else:
        raise ValueError(f"Unknown view kind {kind!r}")
    return written
