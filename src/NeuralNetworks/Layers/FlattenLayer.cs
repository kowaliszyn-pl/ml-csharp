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

    protected override void EnsureSameShapeForInput(float[,,,]? input, float[,,,]? inputGradient)
        => EnsureSameShape(input, inputGradient);

    protected override void EnsureSameShapeForOutput(float[,]? output, float[,]? outputGradient)
        => EnsureSameShape(output, outputGradient);

    public override string ToString()
        => "FlattenLayer";
}
