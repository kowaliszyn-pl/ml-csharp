// Neural Networks in C♯
// File name: Sigmoid.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Sigmoid (Logistic) activation function.
/// </summary>
/// <remarks>
/// <para><b>Formula:</b> f(x) = 1 / (1 + exp(-x)) = σ(x)</para>
/// <para><b>Input Gradient Formula:</b> ∂L/∂x = ∂L/∂y · f(x) · (1 - f(x)) = ∂L/∂y · σ(x) · (1 - σ(x))</para>
/// <para><b>Output Range:</b> (0, 1)</para>
/// <para><b>Description:</b> The sigmoid function is a smooth, S-shaped curve that squashes input values into the range (0, 1). 
/// It is commonly used in binary classification tasks, especially in the output layer for predicting probabilities. 
/// The function is differentiable everywhere, making it suitable for gradient-based optimization methods.</para>
/// <para><b>Remarks:</b> The sigmoid function suffers from vanishing gradient problems for large positive or negative inputs, 
/// as the gradient approaches zero at the extremes. This can slow down training in deep networks. 
/// For hidden layers, alternatives like <see cref="ReLU2D"/> or <see cref="Tanh2D"/> are often preferred. The sigmoid is computationally more expensive than ReLU due to the exponential operation.</para>
/// </remarks>
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
