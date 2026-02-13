// Neural Networks in C♯
// File name: LeakyReLU2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class LeakyReLU2D(float alfa = 0.01f, float beta = 1f) : ActivationFunction<float[,], float[,]>
{
    protected override float[,] CalcOutput(bool inference)
        => LeakyReLUOutput(Input, alfa, beta);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => LeakyReLUInputGradient(outputGradient, Input, alfa, beta);

    public override string ToString()
        => $"LeakyReLU2D (alfa={alfa}, beta={beta})";
}
