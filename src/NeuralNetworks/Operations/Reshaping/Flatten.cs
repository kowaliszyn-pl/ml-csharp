// Neural Networks in C♯
// File name: Flatten.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Reshaping;

public class Flatten : Operation<float[,,,], float[,]>
{
    protected override float[,] CalcOutput(bool inference)
        => Flatten(Input);

    protected override float[,,,] CalcInputGradient(float[,] outputGradient)
        => Unflatten(outputGradient, Input);

    public override string ToString()
        => "Flatten";
}
