# multi\_head\_attention

```text
    description: 'layer multi_head_attention for configurable component'
    ---

    # multi_head_attention layer

    Adds an attention layer with multiple heads. Uses softmax. Ends in a linear

    ## required arguments

        `size` - output size

    ## optional arguments

        `heads` - Number of heads to use. Defaults to 4

    ## input size

    Any 2-d tensor

    ## output size

    First argument

    ## syntax

    ```json
      "ez_norm CHANNELS heads=HEADS"
    ```

    ## examples

    ```json
      "multi_head_attention 1024 heads=4"
    ```
```

