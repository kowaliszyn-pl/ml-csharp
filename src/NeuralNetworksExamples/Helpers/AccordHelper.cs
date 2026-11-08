// Neural Networks in C♯
// File name: AccordHelper.cs
// www.kowaliszyn.pl, 2025 - 2026

using Accord.MachineLearning.Clustering;

namespace NeuralNetworksExamples.Helpers;

internal static class AccordHelper
{
    public static float[,] TransformWithTSNE(float[,] inputData, int outputDimensions = 2, int perplexity = 30)
    {
        // Convert to double[][] for Accord.NET
        int rows = inputData.GetLength(0);
        int inputDimensions = inputData.GetLength(1);
        double[][] inputDataDouble = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            inputDataDouble[i] = new double[inputDimensions];
            for (int j = 0; j < inputDimensions; j++)
            {
                inputDataDouble[i][j] = inputData[i, j];
            }
        }
        TSNE tsne = new()
        {
            NumberOfOutputs = outputDimensions,
            Perplexity = perplexity
        };
        double[][] reducedDouble = tsne.Transform(inputDataDouble);
        float[,] reducedFloat = new float[rows, outputDimensions];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < outputDimensions; j++)
            {
                reducedFloat[i, j] = (float)reducedDouble[i][j];
            }
        }
        return reducedFloat;
    }
}
