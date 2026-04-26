// Neural Networks in C♯
// File name: Softplus.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Softplus is a smooth approximation of the ReLU (Rectified Linear Unit) activation function. It is defined as f(x) = log(1 + exp(x)) and is differentiable everywhere, making it suitable for gradient-based optimization methods.
/// </summary>
public class Softplus : ActivationFunction<float[,], float[,]>
{
    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => SoftplusInputGradient(outputGradient, Output);

    protected override float[,] CalcOutput(bool inference)
        => SoftplusOutput(Input);

    public override string ToString() 
        => "Softplus";
}
