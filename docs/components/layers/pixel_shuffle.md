
        ---
        description: 'layer pixel_shuffle for configurable component'
        ---

        # pixel_shuffle layer

        Implements PixelShuffle https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html

        ## input size

        Any 4-d tensor of the form `[B, C, H, W]`

        ## output size

        A 4d-tensor of the form `[B, C//4, H*2, W*2]`

        ## syntax

        ```json
          "pixel_shuffle"
        ```
    