// Neural Networks in C♯
// File name: FlattenLayer.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations;
using NeuralNetworks.Operations.Reshaping;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Layers;

public class FlattenLayer : Layer<float[,,,], float[,]>
{
    public override OperationListBuilder<float[,,,], float[,]> CreateOperationListBuilder()
        => AddOperation(new Flatten());

    public override string ToString()
        => "FlattenLayer";
}
