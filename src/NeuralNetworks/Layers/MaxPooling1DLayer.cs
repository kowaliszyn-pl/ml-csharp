// Neural Networks in C♯
// File name: MaxPooling3DLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations.Reshaping;

namespace NeuralNetworks.Layers;

public class MaxPooling1DLayer(int size) : Layer<float[,,], float[,,]>
{
    public override OperationListBuilder<float[,,], float[,,]> CreateOperationListBuilder() 
        => AddOperation(new MaxPooling1D(size));
}
