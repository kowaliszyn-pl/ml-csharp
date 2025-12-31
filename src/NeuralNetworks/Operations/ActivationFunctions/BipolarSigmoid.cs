// Neural Networks in C♯
// File name: BipolarSigmoid.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core.Extensions;

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
        => Input.Sigmoid().Add(-0.5f).Multiply(scale);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        Debug.Assert(scale != 0f, "Scale must be non-zero.");

        // Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // Output = scale * (σ(x) - 0.5)  =>  σ(x) = (Output/scale) + 0.5
        // d/dx[scale * (σ(x) - 0.5)] = scale * σ(x) * (1 - σ(x))
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * scale * σ(x) * (1 - σ(x)).
        float[,] sigma = Output.Divide(scale).Add(0.5f);
        float[,] sigmoidBackward = sigma.MultiplyElementwise(sigma.AsOnes().Subtract(sigma)).Multiply(scale);
        return outputGradient.MultiplyElementwise(sigmoidBackward);
    }

    public override string ToString() => $"BipolarSigmoid (scale={scale})";
}
