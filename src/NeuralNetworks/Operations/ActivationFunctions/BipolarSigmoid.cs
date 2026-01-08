// Neural Networks in C♯
// File name: BipolarSigmoid.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Represents a bipolar sigmoid activation operation that applies a scaled sigmoid function shifted to the range [-0.5,
/// 0.5].
/// </summary>
/// <remarks>The bipolar sigmoid function is commonly used in neural networks to introduce non-linearity while
/// centering the output around zero. This operation computes output as scale × (σ(x) – 0.5), where σ(x) is the standard
/// sigmoid function. Setting an appropriate scale can affect the gradient flow and the range of activations.
/// <para>
/// With scale = 2: y = 2σ(x) − 1, which is exactly the bipolar (a.k.a. symmetric/zero-centered) sigmoid. It’s mathematically identical to tanh(x/2). For general scale s: y = (s/2) · tanh(x/2), i.e., a scaled tanh.Range is [−s/2, s/2].
/// </para>
/// </remarks>
/// <param name="scale">The scaling factor applied to the output of the sigmoid function. Must be non-zero.</param>
public class BipolarSigmoid(float scale) : ActivationFunction2D
{
    protected override float[,] CalcOutput(bool inference)
        => BipolarSigmoidOutput(Input, scale);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => BipolarSigmoidInputGradient(outputGradient, Output, scale);

    public override string ToString()
        => $"BipolarSigmoid (scale={scale})";
}
