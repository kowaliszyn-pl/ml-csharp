// Neural Networks in C♯
// File name: BipolarSigmoid.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Bipolar (Symmetric) Sigmoid activation function.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = scale · (σ(x) - 0.5) = scale · (1/(1 + e^(-x)) - 0.5)</para>
/// <para><b>Special case (scale=2):</b> f(x) = 2σ(x) - 1, which is mathematically equivalent to tanh(x/2)</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · scale · σ(x) · (1 - σ(x))</para>
/// <para><b>Output Range:</b> (-scale/2, scale/2)</para>
/// <para><b>Description:</b> A variant of the <see cref="Sigmoid"/> function that centers the output around zero by subtracting 0.5 and applying a scale factor. 
/// This creates a zero-centered activation function, which can improve gradient flow compared to the standard sigmoid. 
/// The scale parameter allows control over the output range, enabling flexibility in network design.</para>
/// <para><b>Remarks:</b> With scale = 2, this produces the classic bipolar sigmoid f(x) = 2σ(x) - 1, ranging from (-1, 1), 
/// which is mathematically identical to tanh(x/2). For general scale s, it produces a scaled tanh: f(x) = (s/2) · tanh(x/2). 
/// The zero-centered nature helps mitigate the zigzag gradient updates that can occur with non-zero-centered activations like standard sigmoid. 
/// However, it still inherits the vanishing gradient problem from the sigmoid function.</para>
/// </remarks>
/// <param name="scale">The scaling factor applied to the output of the sigmoid function. Must be non-zero.</param>
public class BipolarSigmoid(float scale = 1f) : ActivationFunction<float[,], float[,]>
{
    protected override float[,] CalcOutput(bool inference)
        => BipolarSigmoidOutput(Input, scale);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => BipolarSigmoidInputGradient(outputGradient, Output, scale);

    public override string ToString()
        => $"BipolarSigmoid (scale={scale})";
}
