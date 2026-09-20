"""Explicit HyperGAN recovery contract, independent of release provenance.

Bump CURRENT_VERSION when a known change makes saved state or continuation
semantics incompatible. Schema-1 checkpoints written before this field existed
use version 1. Git revisions and package versions identify what ran; they do not
decide whether HyperGAN can restore it. External components and numerical runtime
settings retain their existing strict checks.
"""
import json


CURRENT_VERSION = 1


def validate_checkpoint_compatibility(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('Invalid HyperGAN checkpoint compatibility metadata')
    version = metadata.get('hypergan_checkpoint_version', 1)
    if type(version) is not int or version != CURRENT_VERSION:
        raise ValueError(
            f'Unsupported HyperGAN checkpoint compatibility version: {version!r}; '
            f'this installation supports version {CURRENT_VERSION}')


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def validate_runtime(saved, current):
    if not isinstance(saved, dict) or not isinstance(current, dict):
        raise ValueError('Invalid resume runtime metadata')
    # A HyperGAN release can change without changing its recovery contract.
    saved = {key: value for key, value in saved.items() if key != 'hypergan'}
    current = {key: value for key, value in current.items() if key != 'hypergan'}
    if not _same(saved, current):
        raise ValueError('Resume runtime/topology differs from checkpoint')


def validate_implementation(saved, current):
    for value in (saved, current):
        if not isinstance(value, dict) or any(
                not isinstance(key, str) or not isinstance(digest, str)
                for key, digest in value.items()):
            raise ValueError('Invalid resume implementation metadata')
    def external(value):
        return {key: digest for key, digest in value.items()
                if key != 'hypergan' and not key.startswith('hypergan.')}
    if not _same(external(saved), external(current)):
        raise ValueError('Resume implementation differs from checkpoint: external component or dependency changed')
