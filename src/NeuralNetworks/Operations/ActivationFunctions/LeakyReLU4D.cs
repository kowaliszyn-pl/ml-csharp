// Neural Networks in C♯
// File name: LeakyReLU4D.cs
// www.kowaliszyn.pl, 2025

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class LeakyReLU4D(float alfa = 0.01f, float beta = 1f) : ActivationFunction4D
{
    protected override float[,,,] CalcOutput(bool inference)
        => LeakyReLU(Input, alfa, beta);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient) 
        => LeakyReLUInputGradient(outputGradient, Input, alfa, beta);

    public override string ToString() 
        => $"LeakyReLU4D (alfa={alfa}, beta={beta})";
}
