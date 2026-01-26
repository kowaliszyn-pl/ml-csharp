// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/jaymody/picoGPT (fork https://github.com/kowaliszyn-pl/pico-gpt-2)
// Also, part of the code also copied from https://github.com/lofcz/gpt2sharp (fork https://github.com/kowaliszyn-pl/sharp-gpt-2)

internal sealed record Gpt2Params
{
    public float[,] TokenEmbeddings { get; init; } = default!;
    public float[,] PositionalEmbeddings { get; init; } = default!;
    public Gpt2Block[] Blocks { get; init; } = default!;
    public Gpt2LayerNormParams FinalLayerNorm { get; init; } = default!;
}

internal sealed record Gpt2Block
{
    public Gpt2LayerNormParams LayerNorm1 { get; init; } = default!;
    public Gpt2MultiHeadAttentionParams Attention { get; init; } = default!;
    public Gpt2LayerNormParams LayerNorm2 { get; init; } = default!;
    public Gpt2MultiLayerPerceptron MultiLayerPerceptron { get; init; } = default!;
}

internal sealed record Gpt2LayerNormParams
{
    public float[] Gamma { get; init; } = default!;
    public float[] Beta { get; init; } = default!;
}

internal sealed record Gpt2MultiHeadAttentionParams
{
    public Gpt2LinearParams Projection { get; init; } = default!;
    public Gpt2LinearParams OutputProjection { get; init; } = default!;
}

internal sealed record Gpt2LinearParams
{
    public float[,] Weights { get; init; } = default!;
    public float[] Bias { get; init; } = default!;
}

internal sealed record Gpt2MultiLayerPerceptron
{
    /// <summary>
    /// Multi layer perceptron, full connected
    /// </summary>
    public Gpt2LinearParams FullyConnected { get; init; } = default!;

    /// <summary>
    /// Multi layer perceptron, projection
    /// </summary>
    public Gpt2LinearParams OutputProjection { get; init; } = default!;
}