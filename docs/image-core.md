# Image recipe numerical controls

A component's `trainable = true` preserves the parameter mask and module modes
chosen by its factory. A pretrained feature backbone can remain frozen and in
evaluation mode while its head trains. `trainable = false` freezes the entire
component and sets it to evaluation mode. Frozen parameters still permit input
gradients, including the second backward needed by exact b-cap.

Native and replicated checkpoints retain these masks, nested module modes,
buffers and optimizer ownership. The factory must establish the same parameter
inventory when reconstructing a supported run.
