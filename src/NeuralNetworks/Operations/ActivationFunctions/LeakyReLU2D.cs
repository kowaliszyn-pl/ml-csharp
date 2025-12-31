// Neural Networks in C♯
// File name: LeakyReLU.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core.Extensions;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class LeakyReLU2D(float alfa = 0.01f, float beta = 1f) : ActivationFunction2D
{
    protected override float[,] CalcOutput(bool inference)
        => Input.LeakyReLU(alfa, beta);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        int rows = Input.GetLength(0);
        int cols = Input.GetLength(1);
        float[,] inputGradient = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                inputGradient[i, j] = Input[i, j] > 0 ? outputGradient[i, j] * beta : outputGradient[i, j] * alfa;
            }
        }
        return inputGradient;

    }

    public override string ToString() => $"LeakyReLU2D (alfa={alfa}, beta={beta})";
}
