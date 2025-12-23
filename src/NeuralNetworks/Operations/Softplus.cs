// Neural Networks in C♯
// File name: Softplus.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;

namespace NeuralNetworks.Operations;

public class Softplus : Operation2D
{
    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Softplus function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Softplus function softplus(x) = ln(1 + exp(x)) is σ(x) = 1 / (1 + exp(-x)), which is the Sigmoid function.
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * σ(x).
        // The elementwise multiplication of the output gradient and the derivative of the Softplus function is returned as the input gradient.
        // σ(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient
        float[,] sigmoidBackward = Output.Sigmoid();
        return outputGradient.MultiplyElementwise(sigmoidBackward);
    }

    protected override float[,] CalcOutput(bool inference)
    {
        return Input.Softplus();
    }

    override public string ToString() => "Softplus";
}
