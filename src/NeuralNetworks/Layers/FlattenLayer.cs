// Machine Learning Utils
// File name: FlattenLayer.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed.NeuralNetwork.Operations;

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
}
