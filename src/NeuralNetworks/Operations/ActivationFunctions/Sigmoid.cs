// Neural Networks in C♯
// File name: Sigmoid.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core.Extensions;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Sigmoid activation function.
/// </summary>
public class Sigmoid : ActivationFunction2D
{
    protected override float[,] CalcOutput(bool inference)
        => Input.Sigmoid();

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Sigmoid function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Sigmoid function σ(x) = 1 / (1 + exp(-x)) is σ(x) * (1 - σ(x)).
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * σ(x) * (1 - σ(x)).
        // The elementwise multiplication of the output gradient and the derivative of the Sigmoid function is returned as the input gradient.
        // σ(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient
        float[,] sigmoidBackward = Output.MultiplyElementwise(Output.AsOnes().Subtract(Output));
        return outputGradient.MultiplyElementwise(sigmoidBackward);
    }

    public override string ToString() => "Sigmoid";
}
