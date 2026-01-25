// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

public class Gpt2MultiLayerPerceptron
{
    /// <summary>
    /// Multi layer perceptron, full connected
    /// </summary>
    public Gpt2LinearParams FullyConnected { get; set; }

    /// <summary>
    /// Multi layer perceptron, projection
    /// </summary>
    public Gpt2LinearParams OutputProjection { get; set; }
}