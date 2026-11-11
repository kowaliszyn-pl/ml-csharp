// Neural Networks in C♯
// File name: AccordUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using Accord.MachineLearning.Clustering;
using Accord.Statistics.Analysis;

namespace NeuralNetworksExamples.Helpers;

internal static class AccordUtils
{
    public static float[,] TransformWithTsne(float[,] inputData, int outputDimensions, int perplexity)
    {
        double[][] inputDataDouble = ConvertToDoubleJaggedArray(inputData);

        TSNE tsne = new()
        {
            NumberOfOutputs = outputDimensions,
            Perplexity = perplexity
        };

        double[][] reducedDouble = tsne.Transform(inputDataDouble);

        return ConvertToFloat2DArray(outputDimensions, reducedDouble);
    }

    public static float[,] TransformWithPca(float[,] inputData, int outputDimensions)
    {
        double[][] inputDataDouble = ConvertToDoubleJaggedArray(inputData);

        // pol. Analiza Głównych Składowych
        PrincipalComponentAnalysis pca = new()
        {
            Method = PrincipalComponentMethod.Center
        };

        pca.Learn(inputDataDouble);
        pca.NumberOfOutputs = outputDimensions;
        double[][] reducedDouble = pca.Transform(inputDataDouble);

        return ConvertToFloat2DArray(outputDimensions, reducedDouble);
    }

    private static double[][] ConvertToDoubleJaggedArray(float[,] inputData)
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
        return inputDataDouble;
    }

    private static float[,] ConvertToFloat2DArray(int outputDimensions, double[][] reducedDouble)
    {
        int rows = reducedDouble.Length;
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
