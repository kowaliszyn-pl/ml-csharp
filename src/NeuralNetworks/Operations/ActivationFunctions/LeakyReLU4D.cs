// Neural Networks in C♯
// File name: LeakyReLU.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core.Span;

using static NeuralNetworks.Core.Span.OperationOps;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class LeakyReLU4D(float alfa = 0.01f, float beta = 1f) : ActivationFunction4D
{
    protected override float[,,,] CalcOutput(bool inference)
        => Input.LeakyReLU(alfa, beta);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
    {

        return LeakyReLUCalcInputGradient(outputGradient, Input, alfa, beta);
    }

    public override string ToString() => $"LeakyReLU4D (alfa={alfa}, beta={beta})";
}
