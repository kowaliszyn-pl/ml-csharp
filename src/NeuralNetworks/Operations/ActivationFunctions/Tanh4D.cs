// Neural Networks in C♯
// File name: Tanh4D.cs
// www.kowaliszyn.pl, 2025

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class Tanh4D : ActivationFunction<float[,,,], float[,,,]>
{
    protected override float[,,,] CalcOutput(bool inference)
        => TanhOutput(Input);
    
    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient) 
        => TanhInputGradient(outputGradient, Output);

    public override string ToString() 
        => "Tanh4D";
}
