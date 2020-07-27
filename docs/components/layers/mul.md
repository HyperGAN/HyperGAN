
        ---
        description: 'layer mul for configurable component'
        ---

        # mul layer

        Multiplies two or more layers together. Accepts nested layer definitions.

        ## input size

        Any number of matching tensors

        ## output size

        Same as input size

        ## syntax

        ```json
          "mul [layer]*"
        ```

        ## examples

        ```json
          "mul self (noise)"
        ```
    