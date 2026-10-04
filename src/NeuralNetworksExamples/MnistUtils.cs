// Neural Networks in C♯
// File name: MnistUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

using static System.Console;
using static NeuralNetworks.Core.DataUtils;
using static NeuralNetworksExamples.Drawing;

namespace NeuralNetworksExamples;

internal static class MnistUtils
{
    private const int DigitImageSize = 100; // Size of a saved image in pixels

    internal static void DisplayDigit3PredictionExamples(float[,] yTest, float[,] logits, float[,] testImages, string prefix)
    {
        int[] results = logits.Argmax();

        // We want to show the following examples (indexes in the test set):
        // 1. "3" that was correctly predicted as "3"
        // 2. Not "3" that was correctly predicted as not "3"
        // 3. "3" that was incorrectly predicted as not "3"
        // 4. Not "3" that was incorrectly predicted as "3"

        int actual3Predicted3Index = -1, actualNot3PredictedNot3Index = -1, actualNot3Predicted3Index = -1, actual3PredictedNot3Index = -1;

        int actual3Predicted3Count = 0, actualNot3PredictedNot3Count = 0, actualNot3Predicted3Count = 0, actual3PredictedNot3Count = 0;

        int actual3Predicted3Label = -1, actualNot3PredictedNot3Label = -1, actualNot3Predicted3Label = -1, actual3PredictedNot3Label = -1;

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
                if (actual3Predicted3Index == -1)
                {
                    actual3Predicted3Index = i;
                    actual3Predicted3Label = FindDigit(yTest, i);
                }
                actual3Predicted3Count++;
            }
            else if (!is3Predicted && !is3Actual) // predicted digit is not "3" and actual digit is not "3"
            {
                if (actualNot3PredictedNot3Index == -1)
                {
                    actualNot3PredictedNot3Index = i;
                    actualNot3PredictedNot3Label = FindDigit(yTest, i);
                    correctlyPredictedDigit = results[i];
                }
                actualNot3PredictedNot3Count++;
            }

            // Incorrectly predicted
            else if (!is3Predicted && is3Actual) // predicted digit is not "3" but actual digit is "3"
            {
                if (actual3PredictedNot3Index == -1)
                {
                    actual3PredictedNot3Index = i;
                    actual3PredictedNot3Label = FindDigit(yTest, i);
                    incorrectlyPredictedDigit = results[i];
                }
                actual3PredictedNot3Count++;
            }
            else if (is3Predicted && !is3Actual) // predicted digit is "3" but actual digit is not "3"
            {
                if (actualNot3Predicted3Index == -1)
                {
                    actualNot3Predicted3Index = i;
                    actualNot3Predicted3Label = FindDigit(yTest, i);
                }
                actualNot3Predicted3Count++;
            }
        }

        // Correctly predicted
        SaveMnistPicture(DigitImageSize, actual3Predicted3Index, testImages, $"{prefix}_correctlyPredicted3_its{actual3Predicted3Label}");
        SaveMnistPicture(DigitImageSize, actualNot3PredictedNot3Index, testImages, $"{prefix}_correctlyPredictedNot3_its{actualNot3PredictedNot3Label}");

        // Incorrectly predicted
        SaveMnistPicture(DigitImageSize, actual3PredictedNot3Index, testImages, $"{prefix}_incorrectlyPredictedNot3_its{actual3PredictedNot3Label}");
        SaveMnistPicture(DigitImageSize, actualNot3Predicted3Index, testImages, $"{prefix}_incorrectlyPredicted3_its{actualNot3Predicted3Label}");

        // Print the results
        WriteLine("Examples of predictions vs actual values for the digit \"3\":");

        // Correctly predicted
        WriteLine($"1. \"{actual3Predicted3Label}\" that was correctly predicted as \"3\": index {actual3Predicted3Index}, count {actual3Predicted3Count}");
        WriteLine($"2. \"{actualNot3PredictedNot3Label}\" that was correctly predicted as not \"3\" (but \"{correctlyPredictedDigit}\"): index {actualNot3PredictedNot3Index}, count {actualNot3PredictedNot3Count}");

        // Incorrectly predicted
        WriteLine($"3. \"{actual3PredictedNot3Label}\" that was incorrectly predicted as not \"3\" (but \"{incorrectlyPredictedDigit}\"): index {actual3PredictedNot3Index}, count {actual3PredictedNot3Count}");
        WriteLine($"4. \"{actualNot3Predicted3Label}\" that was incorrectly predicted as \"3\": index {actualNot3Predicted3Index}, count {actualNot3Predicted3Count}");

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
