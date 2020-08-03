# resizable\_stack

```text
    description: 'layer resizable_stack for configurable component'
    ---

    # resizable_stack layer

    `resizable_stack` allows for variable size outputs on the generator. A conv stack is repeated until the output size is reached.

    If you specify "segment_softmax" this repeats the pattern:
      upsample
      normalize(expects style vector named 'w')
      conv ...
      activation(before last layer)

    and ends in:
      segment_softmax output_channels

    ## arguments
        * layer type. Defaults to "segment_softmax"
    ## optional arguments
        * softmax_channels - The number of channels before segment_softmax. Defaults to output_channels * 2 * 5
        * max_channels - The most channels for any conv. Default 256
        * style - the style vector to use. Default "w"
    ## input size

    Any 4-d tensor

    ## output size

    [B, output channels, output height, output width]

    ## syntax

    ```json
      "resizable_stack segment_softmax"
    ```

    ## examples

    ```json
      "identity name=w",
      "linear 8*8*256",
      "relu",
      "resizable_stack segment_softmax",
      "hardtanh"
    ```
```

