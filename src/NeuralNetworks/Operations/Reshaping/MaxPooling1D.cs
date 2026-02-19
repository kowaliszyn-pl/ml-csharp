// Neural Networks in C♯
// File name: MaxPooling3D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Reshaping;

internal class MaxPooling1D(int size) : Operation<float[,,], float[,,]>
{
    private int[,,]? _maxIndices;

    protected override float[,,] CalcOutput(bool inference) 
        => MaxPooling1DOutput(Input, size, out _maxIndices);

    protected override float[,,] CalcInputGradient(float[,,] outputGradient)
        => MaxPooling1DInputGradient(Input, outputGradient, size, _maxIndices);

}
