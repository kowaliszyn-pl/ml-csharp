// Neural Networks in C♯
// File name: Tanh2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Hyperbolic Tangent (Tanh) activation function for 2D tensors.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) = (e^(2x) - 1) / (e^(2x) + 1)</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · (1 - f(x)²) = ∂L/∂y · (1 - tanh²(x))</para>
/// <para><b>Output Range:</b> (-1, 1)</para>
/// <para><b>Description:</b> The hyperbolic tangent function is a smooth, S-shaped curve that maps input values to the range (-1, 1). 
/// Unlike <see cref="Sigmoid"/>, it is zero-centered, which can help with gradient flow during training. 
/// Commonly used in recurrent neural networks (RNNs) and as a hidden layer activation function.</para>
/// <para><b>Remarks:</b> Tanh is essentially a scaled and shifted version of the sigmoid: tanh(x) = 2σ(2x) - 1. 
/// While it addresses the non-zero-centered issue of sigmoid, it still suffers from vanishing gradients for large absolute input values. 
/// The zero-centered output can lead to faster convergence compared to sigmoid in many cases.</para>
/// </remarks>
public class Tanh2D : ActivationFunction<float[,], float[,]>
{
    protected override float[,] CalcOutput(bool inference)
        => TanhOutput(Input);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
         => TanhInputGradient(outputGradient, Output);

    public override string ToString() 
        => "Tanh2D";
}
