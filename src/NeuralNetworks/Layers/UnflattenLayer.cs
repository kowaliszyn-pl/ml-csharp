// Neural Networks in C♯
// File name: UnflattenLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations.Reshaping;

namespace NeuralNetworks.Layers;

public class UnflattenLayer(int channels, int height, int width) : Layer<float[,], float[,,,]>
{
    public override OperationListBuilder<float[,], float[,,,]> CreateOperationListBuilder()
        => AddOperation(new Unflatten(channels, height, width));

    public override string ToString()
        => "UnflattenLayer";
}
