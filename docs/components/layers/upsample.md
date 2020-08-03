
        ---
        description: 'layer upsample for configurable component'
        ---

        # upsample layer

        `upsample` resizes the input tensor to the specified size.

        ## Optional arguments

            * `h` - requested height. defaults to input height * 2
            * `w` - requested width. defaults to input width * 2

        ## input size

        Any 4-d tensor

        ## output size

        [B, input channels, h, w]

        ## syntax

        ```json
          "upsample"
        ```

        ## examples

        ```json
          "upsample w=96 h=96",
          "conv 4",
          "hardtanh"
        ```
    