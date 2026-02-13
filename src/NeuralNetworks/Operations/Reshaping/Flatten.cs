// Neural Networks in C♯
// File name: Flatten.cs
// www.kowaliszyn.pl, 2025

using static NeuralNetworks.Core.ArrayUtils;
using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.Reshaping;

public class Flatten : Operation<float[,,,], float[,]>
{
    protected override float[,] CalcOutput(bool inference) 
        => Flatten(Input);

    protected override float[,,,] CalcInputGradient(float[,] outputGradient) 
        => Unflatten(outputGradient, Input);

    //protected override void EnsureSameShapeForInput(float[,,,]? input, float[,,,] inputGradient)
    //    => EnsureSameShape(input, inputGradient);

    //protected override void EnsureSameShapeForOutput(float[,]? output, float[,] outputGradient)
    //    => EnsureSameShape(output, outputGradient);

    public override string ToString() 
        => "Flatten";
}
