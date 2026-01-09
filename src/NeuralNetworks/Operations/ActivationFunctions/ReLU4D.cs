// Neural Networks in C♯
// File name: ReLU.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class ReLU4D(float beta = 1f) : ActivationFunction4D
{
    protected override float[,,,] CalcOutput(bool inference)
        => ReLUOutput(Input, beta);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
        => ReLUInputGradient(outputGradient, Input, beta);

    public override string ToString()
        => $"ReLU4D (beta={beta})";
}
