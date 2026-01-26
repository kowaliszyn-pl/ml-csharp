// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/jaymody/picoGPT (fork https://github.com/kowaliszyn-pl/pico-gpt-2)
// Also, part of the code also copied from https://github.com/lofcz/gpt2sharp (fork https://github.com/kowaliszyn-pl/sharp-gpt-2)

internal sealed record Gpt2Params
{
    public float[,] TokenEmbeddings { get; set; } = default!;
    public float[,] PositionalEmbeddings { get; set; } = default!;
    public Gpt2Block[] Blocks { get; set; } = default!;
    public Gpt2LayerNormParams FinalLayerNorm { get; set; } = default!;
}

internal sealed record Gpt2Block
{
    public Gpt2LayerNormParams LayerNorm1 { get; set; } = default!;
    public Gpt2LayerNormParams LayerNorm2 { get; set; } = default!;
    public Gpt2MultiHeadAttentionParams Attention { get; set; } = default!;
    public Gpt2MultiLayerPerceptron MultiLayerPerceptron { get; set; } = default!;
}

internal sealed record Gpt2LayerNormParams
{
    public float[] Gamma { get; set; } = default!;
    public float[] Beta { get; set; } = default!;
}

internal sealed record Gpt2MultiHeadAttentionParams
{
    public Gpt2LinearParams Projection { get; set; } = default!;
    public Gpt2LinearParams OutputProjection { get; set; } = default!;
}

internal sealed record Gpt2LinearParams
{
    public float[,] Weights { get; set; } = default!;
    public float[] Bias { get; set; } = default!;
}

internal sealed record Gpt2MultiLayerPerceptron
{
    /// <summary>
    /// Multi layer perceptron, full connected
    /// </summary>
    public Gpt2LinearParams FullyConnected { get; set; } = default!;

    /// <summary>
    /// Multi layer perceptron, projection
    /// </summary>
    public Gpt2LinearParams OutputProjection { get; set; } = default!;
}