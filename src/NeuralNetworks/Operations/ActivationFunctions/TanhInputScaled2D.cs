// Neural Networks in C♯
// File name: TanhScaled2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Scaled Tanh activation: f(x) = tanh(x / scale)
/// </summary>
public class TanhInputScaled2D(float scale) : ActivationFunction<float[,], float[,]>
{

    readonly float _reciprocalScale = 1.0f / scale; // Precompute the inverse of scale for efficiency

    protected override float[,] CalcOutput(bool inference)
        => Input.Multiply(_reciprocalScale).Tanh(); // Multiple by _reciprocalScale is quicker than Divide by scale

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        float[,] deriv = Output
            .AsOnes()
            .Subtract(Output.MultiplyElementwise(Output))
            .Multiply(_reciprocalScale); // Multiple by _reciprocalScale is quicker than Divide by scale

        return outputGradient.MultiplyElementwise(deriv);
    }

    public override string ToString()
        => $"TanhScaled2D (scale={scale})";

}
