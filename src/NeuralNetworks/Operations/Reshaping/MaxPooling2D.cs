// Neural Networks in C♯
// File name: MaxPooling2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Reshaping;

internal class MaxPooling2D(int sizeHeight, int sizeWidth) : Operation<float[,,,], float[,,,]>
{
    // MaxIndexH and MaxIndexW are the indices of the maximum values in the input array for each pooling window.
    // Both numbers are in the range [0, Input.GetLength(2)) and [0, Input.GetLength(3)), respectively.
    // The array is indexed as [batch, channel, outputHeight, outputWidth].
    private (int MaxIndexH, int MaxIndexW)[,,,]? _maxIndices;

    protected override float[,,,] CalcOutput(bool inference)
        => MaxPooling2DOutput(Input, sizeHeight, sizeWidth, out _maxIndices);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
    {
        Debug.Assert(_maxIndices != null, "Expected _maxIndices to be set during CalcOutput, but it was null. This likely means that CalcInputGradient was called before CalcOutput, which is not the intended usage of this operation.");

        return MaxPooling2DInputGradient(Input, outputGradient, _maxIndices);
    }

}
