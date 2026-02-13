// Neural Networks in C♯
// File name: Operation3D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations;

public abstract class Operation3D : Operation<float[,,], float[,,]>
{
    protected override void EnsureSameShapeForInput(float[,,]? input, float[,,] inputGradient)
        => EnsureSameShape(input, inputGradient);

    protected override void EnsureSameShapeForOutput(float[,,]? output, float[,,] outputGradient)
        => EnsureSameShape(output, outputGradient);
}
