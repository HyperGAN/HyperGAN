
        ---
        description: 'layer cat for configurable component'
        ---

        # cat layer

        Concatenate two or more layers together. Accepts nested layer definitions.

        ## input size

        Any number of matching tensors

        ## output size

        Same as input size

        ## syntax

        ```json
          "cat [layer]*"
        ```

        ## examples

        ```json
          "cat self (attention)"
        ```
    