// Neural Networks in C♯
// File name: ReLU.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Operations;

public class ReLU(float alpha = 1f) : Operation2D
{
    protected override float[,] CalcOutput(bool inference)
    {
        int rows = Input.GetLength(0);
        int cols = Input.GetLength(1);
        float[,] output = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                output[i, j] = Math.Max(0, Input[i, j]) * alpha;
            }
        }
        return output;
    }

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        int rows = Input.GetLength(0);
        int cols = Input.GetLength(1);
        float[,] inputGradient = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                inputGradient[i, j] = Input[i, j] > 0 ? outputGradient[i, j] * alpha : 0f;
            }
        }
        return inputGradient;
    }

    public override string ToString() => $"ReLU (alpha={alpha})";
}
