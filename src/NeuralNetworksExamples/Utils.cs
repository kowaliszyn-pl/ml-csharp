// Neural Networks in C♯
// File name: Utils.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

using static System.Console;

namespace NeuralNetworksExamples;

internal static class Utils
{
    public static void DisplayDigit3PredictionExamples(float[,] yTest, float[,] logits)
    {
        int[] results = logits.Argmax();

        // We want to show the following examples (indexes in the test set):
        // 1. "3" that was correctly predicted as "3"
        // 2. Not "3" that was correctly predicted as not "3"
        // 3. "3" that was incorrectly predicted as not "3"
        // 4. Not "3" that was incorrectly predicted as "3"

        int correctlyPredicted3 = -1, correctlyPredictedNot3 = -1, incorrectlyPredicted3 = -1, incorrectlyPredictedNot3 = -1;
        int rows = results.Length;

        for (int i = 0; i < rows; i++)
        {
            bool is3Predicted = results[i] == 3;
            bool is3Actual = yTest[i, 3] == 1f;
            if (is3Predicted && is3Actual && correctlyPredicted3 == -1) // predicted digit is "3" and actual digit is "3"
            {
                correctlyPredicted3 = i;
            }
            else if (!is3Predicted && !is3Actual && correctlyPredictedNot3 == -1) // predicted digit is not "3" and actual digit is not "3"
            {
                correctlyPredictedNot3 = i;
            }
            else if (!is3Predicted && is3Actual && incorrectlyPredicted3 == -1) // predicted digit is not "3" but actual digit is "3"
            {
                incorrectlyPredicted3 = i;
            }
            else if (is3Predicted && !is3Actual && incorrectlyPredictedNot3 == -1) // predicted digit is "3" but actual digit is not "3"
            {
                incorrectlyPredictedNot3 = i;
            }

            if (correctlyPredicted3 != -1 && correctlyPredictedNot3 != -1 && incorrectlyPredicted3 != -1 && incorrectlyPredictedNot3 != -1)
            {
                break; // we found all examples
            }
        }

        // Print the results
        WriteLine("Examples of predictions vs actual values for the digit \"3\":");
        WriteLine($"1. \"3\" that was correctly predicted as \"3\": index {correctlyPredicted3}");
        WriteLine($"2. Not \"3\" that was correctly predicted as not \"3\": index {correctlyPredictedNot3}");
        WriteLine($"3. \"3\" that was incorrectly predicted as not \"3\": index {incorrectlyPredicted3}");
        WriteLine($"4. Not \"3\" that was incorrectly predicted as \"3\": index {incorrectlyPredictedNot3}");
        WriteLine();
    }
}
