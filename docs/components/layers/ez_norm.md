
        ---
        description: 'layer ez_norm for configurable component'
        ---

        # ez_norm layer

        `ez_norm` is a custom normalization technique that uses a conv of the input by a linear projection of a style vector.

        ## Optional arguments

            `style` - The name of the style vector to use. Defaults to "w"

        ## input size

        Any 4-d tensor

        ## output size

        Same as input size

        ## syntax

        ```json
          "ez_norm style=[style vector name]"
        ```

        ## examples

        ```json
          "latent name=w",
          ...
          "cat self (ez_norm style=w)"
        ```
    