// Neural Networks in C♯
// File name: Utils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

using static System.Console;

namespace NeuralNetworksExamples.Autoencoder;

internal static class Utils
{
    internal static void SaveReconstructionComparison(string modelName, int bottleneckDim, float[,] originalImages, float[,] reconstructedImages)
    {
        WriteLine($"Saving original and reconstructed images.");

        int[] selectedImages = [20, 21, 22, 23, 30];

        foreach (int index in selectedImages)
        {
            Drawing.SaveMnistPicture(100, index, originalImages, $"{modelName}_{bottleneckDim}_original_{index}");
            Drawing.SaveMnistPicture(100, index, reconstructedImages, $"{modelName}_{bottleneckDim}_reconstructed_{index}");
        }
    }
}
