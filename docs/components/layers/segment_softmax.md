# segment\_softmax

```text
    description: 'layer segment_softmax for configurable component'
    ---

    # segment_softmax layer

    `segment_softmax` is a custom layer that allows for masking multiple output channels.

    Suppose you have 30 channels and `segment_softmax 3`. First, the 30 channels split into 15/15.
    The first 15 will be used for softmax and multiplied against the second.
    Then each channel is softmaxed, multiplied, and summed.

    So 30 input channels with 3 output channels equate to 5 input channels for each output channel.

    ## input size

    Any 4-d tensor of the shape `[B, C, H, W]`

    ## output size

    [B, OUTPUT_CHANNELS, H, W]

    ## syntax

    ```json
      "segment_softmax OUTPUT_CHANNELS"
    ```

    ## examples

    At the end of the generator for RGB images:

    ```json
      "conv 30",
      "segment_softmax 3",
      "hardtanh"
    ```
```

