
        ---
        description: 'layer layer for configurable component'
        ---

        # layer layer

        `layer` allows you to reference any layer defined in the rest of the network.

        ## arguments

            `layer_name` - The name of the layer to use

        ## Optional arguments

            `upsample` - If true, upsample the layer to the current size

        ## input size

        Any 4-d tensor

        ## output size

        if upsample true, the current input size
        otherwise the layer size

        ## syntax

        ```json
          "layer z"
        ```

        ## examples

        ```json

          "identity name=encoding",
          ...
          "add self (layer encoding upsample=true)"
        ```
    