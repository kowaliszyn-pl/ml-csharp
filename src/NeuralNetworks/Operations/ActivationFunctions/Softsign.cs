// Neural Networks in C♯
// File name: Softsign.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class Softsign : ActivationFunction<float[,], float[,]>
{
    protected override float[,] CalcOutput(bool inference)
        => SoftsignOutput(Input);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => SoftsignInputGradient(outputGradient, Input);

    public override string ToString()
        => "Softsign";
}
