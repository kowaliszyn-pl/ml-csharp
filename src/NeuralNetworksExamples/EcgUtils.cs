// Neural Networks in C♯
// File name: EcgUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using static System.Console;
using static NeuralNetworks.Core.DataUtils;
using static NeuralNetworksExamples.Drawing;

namespace NeuralNetworksExamples;

internal class EcgUtils
{
    private const int EcgChartWidth = 500;
    private const int EcgChartHeight = 210;
    private const int EcgChartMargin = 15;

    internal static void DisplayClassificationPredictionExamples(float[,] yTest, float[,] predictions, float[,] testImages, string prefix)
    {
        // We want to show the following examples (indexes in the test set):
        // 1. A normal case (class 1) that was correctly predicted as normal
        // 2. An abnormal case (class 0) that was correctly predicted as abnormal
        // 3. A normal case (class 1) that was incorrectly predicted as abnormal
        // 4. An abnormal case (class 0) that was incorrectly predicted as normal

        int correctlyPredictedAsNormalIndex = -1, correctlyPredictedAsAbnormalIndex = -1, incorrectlyPredictedAsAbnormalIndex = -1, incorrectlyPredictedAsNormalIndex = -1;
        int correctlyPredictedAsNormalCount = 0, correctlyPredictedAsAbnormalCount = 0, incorrectlyPredictedAsAbnormalCount = 0, incorrectlyPredictedAsNormalCount = 0;
        int rows = predictions.GetLength(0);
        //for (int i = 0; i < rows; i++)
        for (int i = rows - 1; i >= 0; i--)
        {
            bool actualNormalClass = yTest[i, 0] == 1f;
            bool predictedNormalClass = predictions[i, 0] >= 0.5f; // predicted probability of being normal (class 1) is >= 50%

            // A normal case (class 1) that was correctly predicted as normal
            if (predictedNormalClass && actualNormalClass)
            {
                if (correctlyPredictedAsNormalIndex == -1)
                    correctlyPredictedAsNormalIndex = i;
                correctlyPredictedAsNormalCount++;
            }

            // An abnormal case (class 0) that was correctly predicted as abnormal
            else if (!predictedNormalClass && !actualNormalClass)
            {
                if (correctlyPredictedAsAbnormalIndex == -1)
                    correctlyPredictedAsAbnormalIndex = i;
                correctlyPredictedAsAbnormalCount++;
            }

            // A normal case (class 1) that was incorrectly predicted as abnormal
            else if (!predictedNormalClass && actualNormalClass)
            {
                if (incorrectlyPredictedAsAbnormalIndex == -1)
                    incorrectlyPredictedAsAbnormalIndex = i;
                incorrectlyPredictedAsAbnormalCount++;
            }

            // An abnormal case (class 0) that was incorrectly predicted as normal
            else if (predictedNormalClass && !actualNormalClass)
            {
                if (incorrectlyPredictedAsNormalIndex == -1)
                    incorrectlyPredictedAsNormalIndex = i;
                incorrectlyPredictedAsNormalCount++;
            }
        }

        // Correctly predicted
        SaveEcg200Picture(EcgChartWidth, EcgChartHeight, EcgChartMargin, correctlyPredictedAsNormalIndex, testImages, $"{prefix}-correctlyPredictedNormal-its{yTest[correctlyPredictedAsNormalIndex, 0]}");
        SaveEcg200Picture(EcgChartWidth, EcgChartHeight, EcgChartMargin, correctlyPredictedAsAbnormalIndex, testImages, $"{prefix}-correctlyPredictedAbnormal-its{yTest[correctlyPredictedAsAbnormalIndex, 0]}");

        // Incorrectly predicted
        SaveEcg200Picture(EcgChartWidth, EcgChartHeight, EcgChartMargin, incorrectlyPredictedAsAbnormalIndex, testImages, $"{prefix}-incorrectlyPredictedAbnormal-its{yTest[incorrectlyPredictedAsAbnormalIndex, 0]}");
        SaveEcg200Picture(EcgChartWidth, EcgChartHeight, EcgChartMargin, incorrectlyPredictedAsNormalIndex, testImages, $"{prefix}-incorrectlyPredictedNormal-its{yTest[incorrectlyPredictedAsNormalIndex, 0]}");

        // Print the results
        WriteLine("Examples of predictions vs actual values for the test set:");

        // Correctly predicted
        WriteLine($"1. Normal case correctly predicted as normal. {FormatPredictionDetails(correctlyPredictedAsNormalIndex, correctlyPredictedAsNormalCount)}");
        WriteLine($"2. Abnormal case correctly predicted as abnormal. {FormatPredictionDetails(correctlyPredictedAsAbnormalIndex, correctlyPredictedAsAbnormalCount)}");

        // Incorrectly predicted
        WriteLine($"3. Normal case incorrectly predicted as abnormal. {FormatPredictionDetails(incorrectlyPredictedAsAbnormalIndex, incorrectlyPredictedAsAbnormalCount)}");
        WriteLine($"4. Abnormal case incorrectly predicted as normal. {FormatPredictionDetails(incorrectlyPredictedAsNormalIndex, incorrectlyPredictedAsNormalCount)}");

        WriteLine($"The corresponding images have been saved as JPG files in the current bin directory.");

        string FormatPredictionDetails(int index, int count) => $"Index: {index}, predicted probability of being normal: {predictions[index, 0]:P2}, actual class: {(yTest[index, 0] == 1f ? "\'Normal\'" : "\'Abnormal\'")}, count: {count}";
    }

    internal static float[,] GetEcg200TrainData()
        => LoadTsv(Path.Combine(Program.Ecg200DataFolderPath, "ECG200_TRAIN.tsv"));

    internal static float[,] GetEcg200TestData()
        => LoadTsv(Path.Combine(Program.Ecg200DataFolderPath, "ECG200_TEST.tsv"));
}
