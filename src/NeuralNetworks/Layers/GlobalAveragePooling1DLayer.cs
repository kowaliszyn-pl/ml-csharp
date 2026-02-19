// Neural Networks in C♯
// File name: GlobalAveragePooling1DLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations.Reshaping;

namespace NeuralNetworks.Layers;

public class GlobalAveragePooling1DLayer : Layer<float[,,], float[,]>
{
    public override OperationListBuilder<float[,,], float[,]> CreateOperationListBuilder()
        => AddOperation(new GlobalAveragePooling1D());

    public override string ToString()
        => "GlobalAveragePooling3DLayer";
}
