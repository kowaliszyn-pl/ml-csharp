// Neural Networks in C♯
// File name: ReLU.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;

namespace NeuralNetworks.Operations.ActivationFunctions;

public class ReLU(float beta = 1f) : ActivationFunction2D
{
    protected override float[,] CalcOutput(bool inference)
        => Input.ReLU(beta);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        int rows = Input.GetLength(0);
        int cols = Input.GetLength(1);
        float[,] inputGradient = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                inputGradient[i, j] = Input[i, j] > 0 ? outputGradient[i, j] * beta : 0f;
            }
        }
        return inputGradient;
    }

    public override string ToString() => $"ReLU (beta={beta})";
}
