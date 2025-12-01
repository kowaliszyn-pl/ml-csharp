// Machine Learning Utils
// File name: Operation4D.cs
// Code It Yourself with .NET, 2024

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations;

public abstract class Operation4D : Operation<float[,,,], float[,,,]>
{
    protected override void EnsureSameShapeForInput(float[,,,]? input, float[,,,] inputGradient)
        => EnsureSameShape(input, inputGradient);

    protected override void EnsureSameShapeForOutput(float[,,,]? output, float[,,,] outputGradient)
        => EnsureSameShape(output, outputGradient);
}
