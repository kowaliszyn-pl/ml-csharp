// Neural Networks in C♯
// File name: TanhScaled2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class TanhScaled2D(float scale) : ActivationFunction<float[,], float[,]>
{

    protected override float[,] CalcOutput(bool inference)
        => Input.Multiply(1.0f / scale).Tanh(); // Multiple by (1 / scale) is quicker than Divide by scale

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        float[,] deriv = Output
            .AsOnes()
            .Subtract(Output.MultiplyElementwise(Output))
            .Multiply(1.0f / scale); // Multiple by (1 / scale) is quicker than Divide by scale

        return outputGradient.MultiplyElementwise(deriv);
    }

    public override string ToString()
        => $"TanhScaled2D (scale={scale})";

}
