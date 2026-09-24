# Logo source image recovery, 2026-09-24

The 128px logo run failed at step 4176 with a source-content mismatch.
Its manifest records step 4000 as the last durable checkpoint.

Affected source under `/mnt/ml7tb/data/logos256`:
`android/sZAvxZKYwLD45F9Vks4FstOG7MRNVmnlLRC_LyLb8xSQurVFF7qZKQZwNUB4TDr684o=w512.png`.

- Size: 279097 bytes, matching the inventory.
- Expected SHA-256: `30444d54d24a0fc3ebacc8361dfe282def91438a3f32091dcd6cfc12562e4a2e`.
- Observed SHA-256 on three reads: `0b61b52262a3ba6bc2bf11d7ac4be4eb08c37c4a8de3a36ee051c38a2c250638`.
- One IDAT chunk failed its PNG CRC; Pillow `verify()` also rejected it.
- Flipping bit 7 at zero-based byte offset 249861 (0x79 to 0xf9)
  recovered the exact expected SHA-256, and Pillow verification passed.

The PNG chunk CRC identified a candidate single-bit correction; the manifest's
full-file SHA-256 independently confirmed the original bytes. This is an exact
source restoration, with no inventory, split, recipe, or loader changes.
The cause of the bit corruption remains unknown.

The damaged file was backed up, with its original metadata, under
`/home/martyn/dev/hypergan/training-runs/dataset-recovery-2026-09-24/` using
the same basename. An initial in-place write followed by fsync still read back
the damaged hash; the reason was not determined. Recovery then used a new
temporary file in the source directory, verified its hash, and atomically
replaced the source, followed by a directory fsync. The final source hash matched.

Validation in a fresh Python process loaded the unchanged pinned manifest and
decoded the restored entry through both serial and threaded/prefetched paths.
Both produced identical 49152-byte 128x128 RGB payloads. The loader's source
verification remains enabled. Other source files were not audited in this task.
Training was not restarted; the existing launcher can resume the checkpoint.
