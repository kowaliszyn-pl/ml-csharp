// Neural Networks in C♯
// File name: LeakyReLU.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class LeakyReLU4D(float alfa = 0.01f, float beta = 1f) : ActivationFunction4D
{
    protected override float[,,,] CalcOutput(bool inference)
        => Input.LeakyReLU(alfa, beta);

    protected override float[,,,] CalcInputGradient(float[,,,] outputGradient)
    {
        int dim1 = Input.GetLength(0);
        int dim2 = Input.GetLength(1);
        int dim3 = Input.GetLength(2);
        int dim4 = Input.GetLength(3);
        float[,,,] inputGradient = new float[dim1, dim2, dim3, dim4];
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        inputGradient[i, j, k, l] = Input[i, j, k, l] > 0 ? outputGradient[i, j, k, l] * beta : outputGradient[i, j, k, l] * alfa;
                    }
                }
            }
        }
        return inputGradient;

    }

    public override string ToString() => $"LeakyReLU (alfa={alfa}, beta={beta})";
}
