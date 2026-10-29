# Commit message instructions

- Add emoji at the end of your standard (default) commit message.
- The first line should contain:
    - A *title* - a one or multi word (separated by dot ".") description of the change followed by a colon and a space, for example:
        - NeuralNetworks.Optimizer.Refactor: 
        - Examples.Autoencoders.Fix: 
        - NeuralNetworks.Gpt-2.Tokenizer.Optimize: 
    - A *short summary* of the changes made
    - *Emoji* at the end of the line related to the commit message.
- Include a body only when it adds useful context.
- Do not mention files individually unless necessary.
- Never invent changes that are not present in the diff.

# Naming conventions for dimensions in code

- For 1D arrays, use `length`.
- For 2D arrays, use `rows` and `columns` for the first and second dimensions, respectively.
- For 3D arrays and higher-dimensional arrays, use `dim0`, `dim1`, `dim2`, and so on.