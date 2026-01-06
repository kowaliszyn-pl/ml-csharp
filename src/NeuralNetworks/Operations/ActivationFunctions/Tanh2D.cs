// Neural Networks in C♯
// File name: Tanh2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class Tanh2D : ActivationFunction2D
{
    protected override float[,] CalcOutput(bool inference)
        => Tanh(Input);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
         => TanhInputGradient(outputGradient, Output);

    public override string ToString() 
        => "Tanh2D";
}
