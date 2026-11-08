// Neural Networks in C♯
// File name: AccordUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using Accord.MachineLearning.Clustering;
using Accord.Statistics.Analysis;

namespace NeuralNetworksExamples.Helpers;

internal static class AccordUtils
{
    public static float[,] TransformWithTSNE(float[,] inputData, int outputDimensions = 2, int perplexity = 30)
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

    public static float[,] TransformWithPCA(float[,] inputData, int outputDimensions = 2)
    {
        double[][] inputDataDouble = ConvertToDoubleJaggedArray(inputData);

        PrincipalComponentAnalysis pca = new()
        {
            Method = PrincipalComponentMethod.CovarianceMatrix
        };

        pca.Learn(inputDataDouble);
        pca.NumberOfOutputs = outputDimensions;
        double[][] reducedDouble = pca.Transform(inputDataDouble);

        //for (int i = 0; i < coords2D.Length; i++)
        //{
        //    Console.WriteLine($"Embedding {i} -> X: {coords2D[i][0]:F4}, Y: {coords2D[i][1]:F4}");
        //}

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
