// Neural Networks in C♯
// File name: Utils.cs
// www.kowaliszyn.pl, 2025 - 2026

using static System.Console;
using static NeuralNetworksExamples.Helpers.AccordUtils;
using static NeuralNetworksExamples.Helpers.ScottPlotUtils;

namespace NeuralNetworksExamples.Autoencoder;

internal static class Utils
{
    internal static void SaveReconstructionComparison(string modelName, string bottleneckActivationFunction, int bottleneckDim, float[,] originalImages, float[,] reconstructedImages, float[,] randomlyGeneratedImages)
    {
        WriteLine($"Saving original and reconstructed images.");

        int[] selectedImages = [22, 23, 31, 32, 33, 35];

        foreach (int index in selectedImages)
        {
            Helpers.DrawingUtils.SaveMnistPicture(100, index, originalImages, $"{modelName}_{bottleneckActivationFunction}_{bottleneckDim}_original_{index}");
            Helpers.DrawingUtils.SaveMnistPicture(100, index, reconstructedImages, $"{modelName}_{bottleneckActivationFunction}_{bottleneckDim}_reconstructed_{index}");
        }

        for (int i = 0; i < randomlyGeneratedImages.GetLength(0); i++)
        {
            Helpers.DrawingUtils.SaveMnistPicture(100, i, randomlyGeneratedImages, $"{modelName}_{bottleneckActivationFunction}_{bottleneckDim}_random_{i}");
        }
    }

    internal static void VisualizeWithHistogramAndTSNE(string modelName, string bottleneckActivationFunction, float[,] labels, float[,] encoded)
    {
        int rows = encoded.GetLength(0);
        int dim = encoded.GetLength(1);
        string outputPath;

        try
        {
            float[,] reduced = TransformWithTSNE(encoded, outputDimensions: 2, perplexity: 30);

            WriteLine("Creating visualization...");

            outputPath = $"{modelName}_{bottleneckActivationFunction}_{dim}_{rows}_tsne.png";

            SavePointsAsPng(
                reduced,
                labels,
                $"t-SNE Visualization of Latent Space ({modelName}, bottleneck={dim}, points={rows})",
                "t-SNE Component 1",
                "t-SNE Component 2",
                1200,
                900,
                outputPath
            );

            ForegroundColor = ConsoleColor.Green;
            WriteLine($"t-SNE plot saved to {outputPath}");
            ResetColor();
        }
        catch (Exception ex)
        {
            ForegroundColor = ConsoleColor.Red;
            WriteLine($"Error during t-SNE transformation: {ex.Message}");
            ResetColor();
        }

        // Generate a chart with a histogram of the encoded values (float[,] encoded) within the range [-1, 1] and 20 bins

        WriteLine("Creating histogram of encoded values...");

        outputPath = $"{modelName}_{bottleneckActivationFunction}_{dim}_{rows}_histogram.png";

        SaveHistogramAsPng(
            encoded,
            20,
            $"Histogram of Encoded Values ({modelName}, bottleneck={dim})",
            "Encoded Value",
            "Frequency",
            1200,
            900,
            outputPath);

        ForegroundColor = ConsoleColor.Green;
        WriteLine($"Histogram saved to {outputPath}");
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
                randomEncoded[i, j] = (float)(random.NextDouble() * 2 - 1); // Random values in [-1, 1)
            }
        }
        return randomEncoded;
    }
}
