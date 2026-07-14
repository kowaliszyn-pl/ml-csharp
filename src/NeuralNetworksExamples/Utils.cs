// Neural Networks in C♯
// File name: Utils.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;
using static NeuralNetworksExamples.Drawing;

namespace NeuralNetworksExamples;

internal static class Utils
{
    // MNIST
    private const int DigitImageSize = 100; // Size of a saved image in pixels

    // EGC
    private const int EcgChartWidth = 500;
    private const int EcgChartHeight = 210;
    private const int EcgChartMargin = 15;

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

    internal static float[,] GetEcg200TrainData()
        => LoadTsv(Path.Combine(Program.Ecg200DataFolderPath, "ECG200_TRAIN.tsv"));

    internal static float[,] GetEcg200TestData()
        => LoadTsv(Path.Combine(Program.Ecg200DataFolderPath, "ECG200_TEST.tsv"));

    /// <summary>
    /// Split the 2D array into features (all columns except the first one) and labels (the first column), and convert the labels to a one-hot table.
    /// </summary>
    /// <param name="source">The source 2D array.</param>
    /// <returns>A tuple containing the features and one-hot encoded labels.</returns>
    internal static (float[,] Features, float[,] OneHotLabels) SplitFeaturesAndEncodeLabels(float[,] source)
    {
        // Split into features (all columns except the first one) and labels (the first column)

        (float[,] features, float[,] labels) = SplitFeaturesAndLabels(source);

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
    internal static (float[,] Features, float[,] Labels) SplitFeaturesAndLabels(float[,] source)
    {
        // Split into features (all columns except the first one) and labels (the first column)

        float[,] features = ExtractFeatureColumns(source);
        float[,] labels = source.GetColumn(0);

        return (features, labels);
    }

    /// <summary>
    /// Extract the feature columns from the 2D array (all columns except the first one).
    /// </summary>
    /// <param name="source">The source 2D array.</param>
    /// <returns>The extracted feature columns.</returns>
    internal static float[,] ExtractFeatureColumns(float[,] source) 
        => source.GetColumns(1..source.GetLength(1));

    /// <summary>
    /// Extract the feature columns from the 2D array and normalize the pixel values to the range [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1], and reshape to a 4D array with shape (batch, channels, height, width).
    /// </summary>
    /// <param name="source">The source 2D array.</param>
    /// <returns>The extracted and normalized feature columns as a 4D array.</returns>
    internal static float[,,,] ExtractFeaturesAsTanhNormalized4D(float[,] source)
    {
        // Extract the feature columns from the 2D array (all columns except the first one)

        float[,] features = ExtractFeatureColumns(source);

        // Normalize the pixel values to the range [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1], and reshape to a 4D array with shape (batch, channels, height, width)

        return TanhNormalizeAndReshapeTo4D(features);
    }

    /// <summary>
    /// Normalize the pixel values in the 2D array to the range [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1].
    /// </summary>
    /// <param name="source"></param>
    internal static void TanhNormalizeInPlace(float[,] source)
    {
        // Scale to range [-1, 1]
        
        const float min = 0;
        const float max = 255f;
        const float scale = 2f / (max - min); 
        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                source[row, col] = (source[row, col] - min) * scale - 1f;
            }
        }
    }

    /// <summary>
    /// Convert a 2D array of pixel values to a 4D array with shape (batch, channels, height, width) and normalize the pixel values to the range [-1, 1] for better training of the autoencoder with Tanh activation function which outputs values in the range [-1, 1].
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    internal static float[,,,] TanhNormalizeAndReshapeTo4D(float[,] source)
    {
        // Scale to range [-1, 1]

        const float min = 0;
        const float max = 255f;
        const float scale = 2f / (max - min); 

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        float[,,,] res = new float[rows, 1, 28, 28];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                res[row, 0 /* one input channel */, col / 28 /* h */, col % 28 /* w */] = source[row, col] * scale - 1f;
            }
        }

        return res;
    }

    /// <summary>
    /// Denormalize the pixel values in the 2D array from the range [-1, 1] back to the original range [0, 255].
    /// </summary>
    /// <param name="source"></param>
    internal static void DenormalizeToPixelValuesInPlace(float[,] source)
    {
        const float scale = 255f / 2f;
        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                source[row, col] = (source[row, col] + 1f) * scale;
            }
        }
    }


    /// <summary>
    /// Denormalize the pixel values in the 4D array from the range [-1, 1] back to the original range [0, 255] and reshape to a 2D array with shape (batch, height * width).
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    internal static float[,] DenormalizeAndReshapeTo2D(float[,,,] source)
    {
        const float scale = 255f / 2f;
        int batch = source.GetLength(0);
        int channels = source.GetLength(1);
        int height = source.GetLength(2);
        int width = source.GetLength(3);

        float[,] res = new float[batch, height * width];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        res[b, h * width + w] = (source[b, c, h, w] + 1f) * scale;
                    }
                }
            }
        }

        return res;
    }

    /// <summary>
    /// Standardize the 2D array in place by subtracting the mean and dividing by the standard deviation, and return the mean and standard deviation used for standardization.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    internal static (float Mean, float StdDev) StandardizeInPlace(float[,] source)
    {
        float mean = source.Mean();
        source.AddInPlace(-mean);

        float stdDev = source.StdDev();
        source.DivideInPlace(stdDev);

        return (mean, stdDev);
    }

    /// <summary>
    /// Standardize the 2D array in place by subtracting the given mean and dividing by the given standard deviation.
    /// </summary>
    /// <param name="source"></param>
    /// <param name="mean"></param>
    /// <param name="stdDev"></param>
    internal static void ApplyStandardizationInPlace(float[,] source, float mean, float stdDev)
    {
        source.AddInPlace(-mean);
        source.DivideInPlace(stdDev);
    }

    /// <summary>
    /// Standardize the 4D array in place by subtracting the mean and dividing by the standard deviation, and return the mean and standard deviation used for standardization.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    internal static (float Mean, float StdDev) StandardizeInPlace(float[,,,] source)
    {
        float mean = source.Mean();
        source.AddInPlace(-mean);

        float stdDev = source.StdDev();
        source.DivideInPlace(stdDev);

        return (mean, stdDev);
    }

    /// <summary>
    /// Apply standardization to the 4D array in place by subtracting the given mean and dividing by the given standard deviation.
    /// </summary>
    /// <param name="source"></param>
    /// <param name="mean"></param>
    /// <param name="stdDev"></param>
    internal static void ApplyStandardizationInPlace(float[,,,] source, float mean, float stdDev)
    {
        source.AddInPlace(-mean);
        source.DivideInPlace(stdDev);
    }

    /// <summary>
    /// Reshape the 2D array to a 4D array with shape (batch, channels, height, width) where channels = 1, height = cols / 28, width = 28.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    internal static float[,,,] ReshapeTo4D(float[,] source)
    {
        int rows = source.GetLength(0);
        int cols = source.GetLength(1);
        float[,,,] res = new float[rows, 1, cols / 28, 28];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                res[row, 0 /* one input channel */, col / 28 /* h */, col % 28 /* w */] = source[row, col];
            }
        }
        return res;
    }
}
