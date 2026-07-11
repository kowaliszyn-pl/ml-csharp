// Neural Networks in C♯
// File name: MaxPooling2DLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations.Reshaping;

namespace NeuralNetworks.Layers;

public class MaxPooling2DLayer(int sizeHeight, int sizeWidth) : Layer<float[,,,], float[,,,]>
{
    public override OperationListBuilder<float[,,,], float[,,,]> CreateOperationListBuilder()
        => AddOperation(new MaxPooling2D(sizeHeight, sizeWidth));

    public override string ToString()
        => $"MaxPooling2DLayer (sizeHeight={sizeHeight}, sizeWidth={sizeWidth})";
}
