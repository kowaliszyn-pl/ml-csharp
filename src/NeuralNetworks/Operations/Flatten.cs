// Neural Networks in C♯
// File name: Flatten.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core.Span;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations;

public class Flatten : Operation<float[,,,], float[,]>
{
    protected override float[,] CalcOutput(bool inference) 
        => Input.Flatten();

    protected override float[,,,] CalcInputGradient(float[,] outputGradient) 
        => outputGradient.Unflatten(Input);

    protected override void EnsureSameShapeForInput(float[,,,]? input, float[,,,] inputGradient)
        => EnsureSameShape(input, inputGradient);

    protected override void EnsureSameShapeForOutput(float[,]? output, float[,] outputGradient)
        => EnsureSameShape(output, outputGradient);

    public override string ToString() 
        => "Flatten";
}
