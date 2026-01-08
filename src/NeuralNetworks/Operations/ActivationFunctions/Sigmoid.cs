// Neural Networks in C♯
// File name: Sigmoid.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

/// <summary>
/// Sigmoid activation function.
/// </summary>
public class Sigmoid : ActivationFunction2D
{
    protected override float[,] CalcOutput(bool inference)
        => SigmoidOutput(Input);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => SigmoidInputGradient(outputGradient, Output);

    public override string ToString()
        => "Sigmoid";
}
