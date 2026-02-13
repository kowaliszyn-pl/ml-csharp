// Neural Networks in C♯
// File name: Softplus.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class Softplus : ActivationFunction<float[,], float[,]>
{
    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => SoftplusInputGradient(outputGradient, Output);

    protected override float[,] CalcOutput(bool inference)
        => SoftplusOutput(Input);

    public override string ToString() 
        => "Softplus";
}
