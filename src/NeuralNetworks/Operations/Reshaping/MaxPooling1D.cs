// Neural Networks in C♯
// File name: MaxPooling3D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Reshaping;

internal class MaxPooling1D(int size) : Operation<float[,,], float[,,]>
{
    private int[,,]? _maxIndices;

    protected override float[,,] CalcOutput(bool inference) 
        => MaxPooling1DOutput(Input, size, out _maxIndices);

    protected override float[,,] CalcInputGradient(float[,,] outputGradient)
    {
        Debug.Assert(_maxIndices != null, "Expected _maxIndices to be set during CalcOutput, but it was null. This likely means that CalcInputGradient was called before CalcOutput, which is not the intended usage of this operation.");

        return MaxPooling1DInputGradient(Input, outputGradient, size, _maxIndices);
    }

}
