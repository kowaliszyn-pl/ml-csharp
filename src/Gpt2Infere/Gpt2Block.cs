// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

public class Gpt2Block
{
    public Gpt2LayerNormParams LayerNorm1 { get; internal set; }
    public Gpt2LayerNormParams LayerNorm2 { get; internal set; }
    public Gpt2MultiHeadAttentionParams Attention { get; internal set; }
    public Gpt2MultiLayerPerceptron MultiLayerPerceptron { get; internal set; }
}