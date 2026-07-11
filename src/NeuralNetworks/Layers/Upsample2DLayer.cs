// Neural Networks in C♯
// File name: Upsample2DLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations.Reshaping;

namespace NeuralNetworks.Layers;

public class Upsample2DLayer(int scaleHeight, int scaleWidth) : Layer<float[,,,], float[,,,]>
{
    public override OperationListBuilder<float[,,,], float[,,,]> CreateOperationListBuilder()
        => AddOperation(new Upsample2D(scaleHeight, scaleWidth));

    public override string ToString()
        => $"Upsample2DLayer (scaleHeight={scaleHeight}, scaleWidth={scaleWidth})";
}
