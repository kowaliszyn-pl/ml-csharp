// Neural Networks in C♯
// File name: Tanh4D.cs
// www.kowaliszyn.pl, 2025

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Hyperbolic Tangent (Tanh) activation function for 4D tensors.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) = (e^(2x) - 1) / (e^(2x) + 1)</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · (1 - f(x)²) = ∂L/∂y · (1 - tanh²(x))</para>
/// <para><b>Output Range:</b> (-1, 1)</para>
/// <para><b>Description:</b> The hyperbolic tangent function is a smooth, S-shaped curve that maps input values to the range (-1, 1). 
/// This variant operates on 4D tensors, commonly used in convolutional neural networks (CNNs) where the tensor dimensions typically represent 
/// [batch, channels, height, width].</para>
/// <para><b>Remarks:</b> Functionally identical to <see cref="Tanh2D"/> but designed for 4D tensor operations typical in CNNs. 
/// Maintains the same mathematical properties including zero-centered outputs and susceptibility to vanishing gradients. 
/// The 4D implementation allows for efficient batch processing of multi-channel image data.</para>
/// </remarks>
public class Tanh4D : ActivationFunction<float[,,,], float[,,,]>
{
    protected override float[,,,] CalcOutput(bool inference)
        => TanhOutput(Input);
    
    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient) 
        => TanhInputGradient(outputGradient, Output);

    public override string ToString() 
        => "Tanh4D";
}
