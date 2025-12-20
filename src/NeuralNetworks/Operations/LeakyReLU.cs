// Neural Networks in C♯
// File name: LeakyReLU.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;

namespace NeuralNetworks.Operations;

public class LeakyReLU(float negativeAlpha = 0.5f, float alpha = 1f) : Operation2D
{
    protected override float[,] CalcOutput(bool inference)
        => Input.LeakyReLU(negativeAlpha, alpha);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        int rows = Input.GetLength(0);
        int cols = Input.GetLength(1);
        float[,] inputGradient = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                inputGradient[i, j] = Input[i, j] > 0 ? outputGradient[i, j] * alpha : outputGradient[i, j] * negativeAlpha;
            }
        }
        return inputGradient;

    }

    public override string ToString() => $"LeakyReLU (negativeAlpha={negativeAlpha}, alpha={alpha})";
}
