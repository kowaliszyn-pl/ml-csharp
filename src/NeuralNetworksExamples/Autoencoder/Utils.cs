// Neural Networks in C♯
// File name: Utils.cs
// www.kowaliszyn.pl, 2025 - 2026

using Accord.MachineLearning.Clustering;

using ScottPlot;
using ScottPlot.Plottables;

using static System.Console;

namespace NeuralNetworksExamples.Autoencoder;

internal static class Utils
{
    internal static void SaveReconstructionComparison(string modelName, int bottleneckDim, float[,] originalImages, float[,] reconstructedImages, float[,] randomlyGeneratedImages)
    {
        WriteLine($"Saving original and reconstructed images.");

        int[] selectedImages = [22, 23, 31, 32, 33, 35];

        foreach (int index in selectedImages)
        {
            Drawing.SaveMnistPicture(100, index, originalImages, $"{modelName}_{bottleneckDim}_original_{index}");
            Drawing.SaveMnistPicture(100, index, reconstructedImages, $"{modelName}_{bottleneckDim}_reconstructed_{index}");
            
        }

        for(int i = 0; i < randomlyGeneratedImages.GetLength(0); i++ )
        {
            Drawing.SaveMnistPicture(100, i, randomlyGeneratedImages, $"{modelName}_{bottleneckDim}_random_{i}");
        }
    }

    internal static void VisualizeWithTSNE(string modelName, float[,] labels, float[,] encoded)
    {
        // Convert to double[][] for Accord.NET
        int n = encoded.GetLength(0);
        int dim = encoded.GetLength(1);
        double[][] encodedDouble = new double[n][];
        for (int i = 0; i < n; i++)
        {
            encodedDouble[i] = new double[dim];
            for (int j = 0; j < dim; j++)
            {
                encodedDouble[i][j] = encoded[i, j];
            }
        }

        // Apply t-SNE
        WriteLine("Applying t-SNE reduction...");
        TSNE tsne = new()
        {
            NumberOfOutputs = 2,
            Perplexity = 30
        };

        double[][] reduced = tsne.Transform(encodedDouble);

        // Create plot
        WriteLine("Creating visualization...");
        Plot plt = new();

        // Group points by digit
        for (int digit = 0; digit <= 9; digit++)
        {
            List<double> xPoints = [];
            List<double> yPoints = [];

            for (int i = 0; i < n; i++)
            {
                if ((int)labels[i, 0] == digit)
                {
                    xPoints.Add(reduced[i][0]);
                    yPoints.Add(reduced[i][1]);
                }
            }

            Scatter scatter = plt.Add.ScatterPoints(xPoints, yPoints);
            scatter.LegendText = $"Digit {digit}";
            scatter.MarkerSize = 5;
        }

        plt.ShowLegend();
        plt.Title($"t-SNE Visualization of Latent Space (bottleneck={dim}, points={n})");
        plt.XLabel("t-SNE Component 1");
        plt.YLabel("t-SNE Component 2");

        string outputPath = $"{modelName}_{dim}_{n}_tsne.png";
        plt.SavePng(outputPath, 1200, 900);

        ForegroundColor = ConsoleColor.Green;
        WriteLine($"t-SNE plot saved to {outputPath}");
        ResetColor();
    }

    internal static string GetFileName(string modelName, int bottleneckDim)
        => $"{modelName}_{bottleneckDim}.json";

    internal static float[,] GenerateRandomEncodedData(int bottleneckDim, int imageCount, int randomSeed)
    {
        Random random = new(randomSeed);
        float[,] randomEncoded = new float[imageCount, bottleneckDim];
        for (int i = 0; i < imageCount; i++)
        {
            for (int j = 0; j < bottleneckDim; j++)
            {
                randomEncoded[i, j] = (float)(random.NextDouble() * 2 - 1); // Random values in [-1, 1]
            }
        }
        return randomEncoded;
    }
}
