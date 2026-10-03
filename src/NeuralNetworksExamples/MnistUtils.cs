// Neural Networks in C♯
// File name: Utils.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

using static System.Console;
using static NeuralNetworks.Core.DataUtils;
using static NeuralNetworksExamples.Drawing;

namespace NeuralNetworksExamples;

internal static class MnistUtils
{
    // MNIST
    private const int DigitImageSize = 100; // Size of a saved image in pixels

    internal static void DisplayDigit3PredictionExamples(float[,] yTest, float[,] logits, float[,] testImages, string prefix)
    {
        int[] results = logits.Argmax();

        // We want to show the following examples (indexes in the test set):
        // 1. "3" that was correctly predicted as "3"
        // 2. Not "3" that was correctly predicted as not "3"
        // 3. "3" that was incorrectly predicted as not "3"
        // 4. Not "3" that was incorrectly predicted as "3"

        int correctlyPredicted3Index = -1, correctlyPredictedNot3Index = -1, incorrectlyPredicted3Index = -1, incorrectlyPredictedNot3Index = -1;
        int correctlyPredicted3Count = 0, correctlyPredictedNot3Count = 0, incorrectlyPredicted3Count = 0, incorrectlyPredictedNot3Count = 0;
        int correctlyPredicted3Label = -1, correctlyPredictedNot3Label = -1, incorrectlyPredicted3Label = -1, incorrectlyPredictedNot3Label = -1;
        int correctlyPredictedDigit = -1, incorrectlyPredictedDigit = -1;
        int rows = results.Length;

        for (int i = 0; i < rows; i++)
        //for (int i = rows - 1; i >= 0; i--) // reverse order to show the last examples in the test set
        {
            bool is3Predicted = results[i] == 3;
            bool is3Actual = yTest[i, 3] == 1f;

            // Correctly predicted
            if (is3Predicted && is3Actual) // predicted digit is "3" and actual digit is "3"
            {
                if (correctlyPredicted3Index == -1)
                {
                    correctlyPredicted3Index = i;
                    correctlyPredicted3Label = FindDigit(yTest, i);
                }
                correctlyPredicted3Count++;
            }
            else if (!is3Predicted && !is3Actual) // predicted digit is not "3" and actual digit is not "3"
            {
                if (correctlyPredictedNot3Index == -1)
                {
                    correctlyPredictedNot3Index = i;
                    correctlyPredictedNot3Label = FindDigit(yTest, i);
                    correctlyPredictedDigit = results[i];
                }
                correctlyPredictedNot3Count++;
            }

            // Incorrectly predicted
            else if (!is3Predicted && is3Actual) // predicted digit is not "3" but actual digit is "3"
            {
                if (incorrectlyPredictedNot3Index == -1)
                {
                    incorrectlyPredictedNot3Index = i;
                    incorrectlyPredictedNot3Label = FindDigit(yTest, i);
                    incorrectlyPredictedDigit = results[i];
                }
                incorrectlyPredictedNot3Count++;
            }
            else if (is3Predicted && !is3Actual) // predicted digit is "3" but actual digit is not "3"
            {
                if (incorrectlyPredicted3Index == -1)
                {
                    incorrectlyPredicted3Index = i;
                    incorrectlyPredicted3Label = FindDigit(yTest, i);
                }
                incorrectlyPredicted3Count++;
            }
        }

        // Correctly predicted
        SaveMnistPicture(DigitImageSize, correctlyPredicted3Index, testImages, $"{prefix}_correctlyPredicted3_its{correctlyPredicted3Label}");
        SaveMnistPicture(DigitImageSize, correctlyPredictedNot3Index, testImages, $"{prefix}_correctlyPredictedNot3_its{correctlyPredictedNot3Label}");

        // Incorrectly predicted
        SaveMnistPicture(DigitImageSize, incorrectlyPredictedNot3Index, testImages, $"{prefix}_incorrectlyPredictedNot3_its{incorrectlyPredictedNot3Label}");
        SaveMnistPicture(DigitImageSize, incorrectlyPredicted3Index, testImages, $"{prefix}_incorrectlyPredicted3_its{incorrectlyPredicted3Label}");

        // Print the results
        WriteLine("Examples of predictions vs actual values for the digit \"3\":");

        // Correctly predicted
        WriteLine($"1. \"{correctlyPredicted3Label}\" that was correctly predicted as \"3\": index {correctlyPredicted3Index}, count {correctlyPredicted3Count}");
        WriteLine($"2. \"{correctlyPredictedNot3Label}\" that was correctly predicted as not \"3\" (but \"{correctlyPredictedDigit}\"): index {correctlyPredictedNot3Index}, count {correctlyPredictedNot3Count}");

        // Incorrectly predicted
        WriteLine($"3. \"{incorrectlyPredictedNot3Label}\" that was incorrectly predicted as not \"3\" (but \"{incorrectlyPredictedDigit}\"): index {incorrectlyPredictedNot3Index}, count {incorrectlyPredicted3Count}");
        WriteLine($"4. \"{incorrectlyPredicted3Label}\" that was incorrectly predicted as \"3\": index {incorrectlyPredicted3Index}, count {incorrectlyPredictedNot3Count}");

        WriteLine($"The corresponding images have been saved as JPG files in the current bin directory.");
        WriteLine();
    }

    private static int FindDigit(float[,] yTest, int row)
    {
        for (int digit = 0; digit < 10; digit++)
        {
            if (yTest[row, digit] == 1f)
            {
                return digit;
            }
        }
        throw new Exception("No 1 found in the row.");
    }

    internal static float[,] GetMnistTrainData()
        => LoadCsv(Path.Combine(Program.MnistDataFolderPath, "mnist_train_small.csv"));

    internal static float[,] GetMnistTestData()
        => LoadCsv(Path.Combine(Program.MnistDataFolderPath, "mnist_test.csv"));

    /// <summary>
    /// Split the 2D array into features (all columns except the first one) and labels (the first column), and convert the labels to a one-hot table.
    /// </summary>
    /// <param name="source">The source 2D array.</param>
    /// <returns>A tuple containing the features and one-hot encoded labels.</returns>
    internal static (float[,] Features, float[,] OneHotLabels) SplitFeaturesAndEncodeLabels(this float[,] source)
    {
        // Split into features (all columns except the first one) and labels (the first column)

        (float[,] features, float[,] labels) = source.SplitFeaturesAndLabels();

        // Convert labels to a one-hot table

        float[,] oneHot = new float[labels.GetLength(0), 10];
        for (int row = 0; row < labels.GetLength(0); row++)
        {
            int value = Convert.ToInt32(labels[row, 0]);
            oneHot[row, value] = 1f;
        }

        return (features, oneHot);
    }

    /// <summary>
    /// Split the 2D array into features (all columns except the first one) and labels (the first column).
    /// </summary>
    /// <param name="source">The source 2D array.</param>
    /// <returns>A tuple containing the features and labels.</returns>
    internal static (float[,] Features, float[,] Labels) SplitFeaturesAndLabels(this float[,] source)
    {
        // Split into features (all columns except the first one) and labels (the first column)

        float[,] features = source.ExtractFeatureColumns();
        float[,] labels = source.ExtractLabelColumn();

        return (features, labels);
    }

    /// <summary>
    /// Extract the feature columns from the 2D array (all columns except the first one).
    /// </summary>
    /// <param name="source">The source 2D array.</param>
    /// <returns>The extracted feature columns.</returns>
    internal static float[,] ExtractFeatureColumns(this float[,] source) 
        => source.GetColumns(1..source.GetLength(1));

    internal static float[,] ExtractLabelColumn(this float[,] source)
        => source.GetColumn(0);
}
