// Neural Networks in C♯
// File name: Utils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics.CodeAnalysis;
using System.Drawing;

using NeuralNetworks.Core;

using static System.Console;

namespace NeuralNetworksExamples;

[SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "Display one warning at begining of the method")]
internal static class Utils
{
#warning The Utils class uses System.Drawing, which may not be fully supported on all platforms. Ensure that the necessary dependencies are available and that the application is run in an environment that supports System.Drawing (e.g., Windows).

    private const int DigitImageSize = 100; // Size of the saved image in pixels
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
                if(correctlyPredictedAsNormalIndex == -1)
                    correctlyPredictedAsNormalIndex = i;
                correctlyPredictedAsNormalCount++;
            }
            
            // An abnormal case (class 0) that was correctly predicted as abnormal
            else if (!predictedNormalClass && !actualNormalClass)
            {
                if(correctlyPredictedAsAbnormalIndex == -1)
                    correctlyPredictedAsAbnormalIndex = i;
                correctlyPredictedAsAbnormalCount++;
            }
            
            // A normal case (class 1) that was incorrectly predicted as abnormal
            else if (!predictedNormalClass && actualNormalClass)
            {
                if(incorrectlyPredictedAsAbnormalIndex == -1)
                    incorrectlyPredictedAsAbnormalIndex = i;
                incorrectlyPredictedAsAbnormalCount++;
            }
            
            // An abnormal case (class 0) that was incorrectly predicted as normal
            else if (predictedNormalClass && !actualNormalClass)
            {
                if(incorrectlyPredictedAsNormalIndex == -1)
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
        WriteLine();

        string FormatPredictionDetails(int index, int count)
        {
            return $"Index: {index}, predicted probability of being normal: {predictions[index, 0]:P2}, actual class: {(yTest[index, 0] == 1f ? "\'Normal\'" : "\'Abnormal\'")}, count: {count}";
        }
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

    #region Create and save pictures

    /// <summary>
    /// Saves a single MNIST image (28 * 28) represented by the specified data to a JPG file (<paramref name="size"/> *
    /// <paramref name="size"/> pixels) in the current directory and returns the file path.
    /// </summary>
    /// <remarks>
    /// The method assumes that the input image data size is 28 * 28 = 784. The output image will be resized to the
    /// specified size (<paramref name="size"/> * <paramref name="size"/> pixels) for better visibility. The pixel
    /// values in the input data are expected to be in the range of 0 to 255, where 0 represents black and 255
    /// represents white. The method will create a grayscale image based on these pixel values and save it as a JPG
    /// file.
    /// </remarks>
    /// <param name="index">
    /// The zero-based index (0..imageCount) of the image to save.
    /// </param>
    /// <param name="mnistData">
    /// A two-dimensional array of size (imageCount, 784) containing the pixel values of the MNIST image. Each element
    /// represents a grayscale intensity value from 0 to 255.
    /// </param>
    /// <returns>The full file path of the saved JPG image file.</returns>
    private static string SaveMnistPicture(int size, int index, float[,] mnistData, string fileName)
    {
        float[] imageData = mnistData.GetRow(index);

        // Build the image from the pixel data
        using Bitmap originalBitmap = new(28, 28, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                int i = (y * 28) + x;
                int value = (int)Math.Clamp(imageData[i], 0f, 255f);
                Color color = Color.FromArgb(value, value, value);
                originalBitmap.SetPixel(x, y, color);
            }
        }

        using Bitmap resizedBitmap = new(size, size, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
        using Graphics graphics = Graphics.FromImage(resizedBitmap);
        graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
        graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;
        graphics.DrawImage(originalBitmap, 0, 0, size, size);

        fileName = $"mnist_image_{fileName}.jpg";
        string filePath = Path.Combine(Directory.GetCurrentDirectory(), fileName);
        resizedBitmap.Save(filePath, System.Drawing.Imaging.ImageFormat.Jpeg);

        return filePath;
    }

    private static string SaveEcg200Picture(int chartWidth, int chartHeight, int margin, int index, float[,] ecgData, string fileName)
    {
        float[] chartData = ecgData.GetRow(index);

        // Build the ECG chart from the data
        using Bitmap bitmap = new(chartWidth, chartHeight, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

        using Graphics graphics = Graphics.FromImage(bitmap);
        graphics.Clear(Color.White);

        // Define the drawing area (canvas) inside the margins
        int canvasLeft = margin;
        int canvasTop = margin;
        int canvasWidth = chartWidth - 2 * margin;
        int canvasHeight = chartHeight - 2 * margin;

        // Find the min and max values in the data to scale the chart
        float minValue = chartData.Min();
        float maxValue = chartData.Max();

        // Draw some horizontal and vertical grid lines for better visibility
        using Pen gridPen = new(Color.LightGray, 1);
        for (int i = 0; i < 11; i++)
        {
            float y = canvasTop + canvasHeight * i / 10f;
            graphics.DrawLine(gridPen, canvasLeft, y, canvasLeft + canvasWidth, y);
        }
        for (int i = 0; i < 11; i++)
        {
            float x = canvasLeft + canvasWidth * i / 10f;
            graphics.DrawLine(gridPen, x, canvasTop, x, canvasTop + canvasHeight);
        }

        // Draw the ECG line
        using Pen redPen = new(Color.Red, 2);

        float xStep = canvasWidth / (float)(chartData.Length - 1);
        float yScale = (maxValue - minValue) == 0 ? 1 : canvasHeight / (maxValue - minValue);
        int canvasBottom = canvasTop + canvasHeight;

        // Draw lines between consecutive points
        for (int i = 1; i < chartData.Length; i++)
        {
            float x1 = canvasLeft + (i - 1) * xStep;
            float y1 = canvasBottom - ((chartData[i - 1] - minValue) * yScale);
            float x2 = canvasLeft + i * xStep;
            float y2 = canvasBottom - ((chartData[i] - minValue) * yScale);
            graphics.DrawLine(redPen, x1, y1, x2, y2);
        }

        fileName = $"ecg200-chart-{fileName}.jpg";
        string filePath = Path.Combine(Directory.GetCurrentDirectory(), fileName);
        bitmap.Save(filePath, System.Drawing.Imaging.ImageFormat.Jpeg);

        return filePath;
    }

    #endregion
}
