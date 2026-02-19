// Neural Networks in C♯
// File name: GlobalAveragePooling1D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Reshaping;

internal class GlobalAveragePooling1D : Operation<float[,,], float[,]>
{
    protected override float[,,] CalcInputGradient(float[,] outputGradient)
        => GlobalAveragePooling1DInputGradient(Input, outputGradient);

    protected override float[,] CalcOutput(bool inference)
        => GlobalAveragePooling1DOutput(Input);
}
