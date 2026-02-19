// Neural Networks in C♯
// File name: ReLU3D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class ReLU3D(float beta = 1f) : ActivationFunction<float[,,], float[,,]>
{
    protected override float[,,] CalcInputGradient(float[,,] outputGradient)
        => ReLUInputGradient(outputGradient, Input, beta);

    protected override float[,,] CalcOutput(bool inference)
        => ReLUOutput(Input, beta);

    public override string ToString()
        => $"ReLU3D (beta={beta})";
}
