// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

public class Gpt2MultiHeadAttentionParams
{
    public Gpt2LinearParams Projection { get; internal set; }
    public Gpt2LinearParams OutputProjection { get; internal set; }
}