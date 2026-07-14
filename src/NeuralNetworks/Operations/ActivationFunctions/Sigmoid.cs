// Neural Networks in C♯
// File name: Sigmoid.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Sigmoid activation function. Generates outputs in the range (0, 1) and is commonly used in binary classification tasks. It is defined as f(x) = 1 / (1 + exp(-x)) and is differentiable, making it suitable for gradient-based optimization methods.
/// </summary>
public class Sigmoid : ActivationFunction<float[,], float[,]>
{
    /// <summary>
    /// Calculates the output of the Sigmoid activation function. The range is (0, 1).
    /// </summary>
    /// <param name="inference">Indicates whether the calculation is for inference or training.</param>
    /// <returns>A two-dimensional array of floating-point values representing the output of the Sigmoid activation function.</returns>
    protected override float[,] CalcOutput(bool inference)
        => SigmoidOutput(Input);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => SigmoidInputGradient(outputGradient, Output);

    public override string ToString()
        => "Sigmoid";
}
