// Neural Networks in C♯
// File name: Utils.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

using static System.Console;

namespace NeuralNetworksExamples;

internal static class Utils
{
    public static void DisplayDigit3PredictionExamples(float[,] yTest, float[,] logits, float[,] testImages)
    {
        int[] results = logits.Argmax();

        // We want to show the following examples (indexes in the test set):
        // 1. "3" that was correctly predicted as "3"
        // 2. Not "3" that was correctly predicted as not "3"
        // 3. "3" that was incorrectly predicted as not "3"
        // 4. Not "3" that was incorrectly predicted as "3"

        int correctlyPredicted3Index = -1, correctlyPredictedNot3Index = -1, incorrectlyPredicted3Index = -1, incorrectlyPredictedNot3Index = -1;
        int correctlyPredicted3Label = -1, correctlyPredictedNot3Label = -1, incorrectlyPredicted3Label = -1, incorrectlyPredictedNot3Label = -1;
        int predicited = -1;
        int rows = results.Length;

        for (int i = 0; i < rows; i++)
        {
            bool is3Predicted = results[i] == 3;
            bool is3Actual = yTest[i, 3] == 1f;

            // Correctly predicted
            if (is3Predicted && is3Actual && correctlyPredicted3Index == -1) // predicted digit is "3" and actual digit is "3"
            {
                correctlyPredicted3Index = i;
                correctlyPredicted3Label = FindDigit(yTest, i);
            }
            else if (!is3Predicted && !is3Actual && correctlyPredictedNot3Index == -1) // predicted digit is not "3" and actual digit is not "3"
            {
                correctlyPredictedNot3Index = i;
                correctlyPredictedNot3Label = FindDigit(yTest, i);
            }

            // Incorrectly predicted
            else if (!is3Predicted && is3Actual && incorrectlyPredictedNot3Index == -1) // predicted digit is not "3" but actual digit is "3"
            {
                incorrectlyPredictedNot3Index = i;
                incorrectlyPredictedNot3Label = FindDigit(yTest, i);
                predicited = results[i];
            }
            else if (is3Predicted && !is3Actual && incorrectlyPredicted3Index == -1) // predicted digit is "3" but actual digit is not "3"
            {
                incorrectlyPredicted3Index = i;
                incorrectlyPredicted3Label = FindDigit(yTest, i);
            }

            if (correctlyPredicted3Index != -1 && correctlyPredictedNot3Index != -1 && incorrectlyPredicted3Index != -1 && incorrectlyPredictedNot3Index != -1)
            {
                break; // we found all examples
            }
        }

        // Correctly predicted
        SaveMnistPicture(200, correctlyPredicted3Index, testImages, $"correctlyPredicted3_its{correctlyPredicted3Label}");
        SaveMnistPicture(200, correctlyPredictedNot3Index, testImages, $"correctlyPredictedNot3_its{correctlyPredictedNot3Label}");

        // Incorrectly predicted
        SaveMnistPicture(200, incorrectlyPredictedNot3Index, testImages, $"incorrectlyPredictedNot3_its{incorrectlyPredictedNot3Label}");
        SaveMnistPicture(200, incorrectlyPredicted3Index, testImages, $"incorrectlyPredicted3_its{incorrectlyPredicted3Label}");

        // Print the results
        WriteLine("Examples of predictions vs actual values for the digit \"3\":");

        // Correctly predicted
        WriteLine($"1. \"{correctlyPredicted3Label}\" that was correctly predicted as \"3\": index {correctlyPredicted3Index}");
        WriteLine($"2. \"{correctlyPredictedNot3Label}\" that was correctly predicted as not \"3\": index {correctlyPredictedNot3Index}");

        // Incorrectly predicted
        WriteLine($"3. \"{incorrectlyPredictedNot3Label}\" that was incorrectly predicted as not \"3\" (but \"{predicited}\"): index {incorrectlyPredictedNot3Index}");
        WriteLine($"4. \"{incorrectlyPredicted3Label}\" that was incorrectly predicted as \"3\": index {incorrectlyPredicted3Index}");

        WriteLine($"The corresponding images have been saved as JPG files in the current bin directory.");
        WriteLine();
    }

    private static int FindDigit(float[,] yTest, int row)
    {
        for (int digit = 0; digit < 10; digit++)
        {
            if(yTest[row, digit] == 1f)
            {
                return digit;
            }
        }
        throw new Exception("No 1 found in the row.");
    }

    /// <summary>
    /// Saves a single MNIST image (28 * 28) represented by the specified data to a JPG file (<paramref name="size"/> *
    /// <paramref name="size"/> pixels) in the current directory with a name based on the provided index and returns the
    /// file path.
    /// </summary>
    /// <remarks>
    /// The method assumes that the input image data size is 28 * 28 = 784. The output image will be resized to the
    /// specified size (<paramref name="size"/> * <paramref name="size"/> pixels) for better visibility. The pixel
    /// values in the input data are expected to be in the range of 0 to 255, where 0 represents black and 255
    /// represents white. The method will create a grayscale image based on these pixel values and save it as a JPG
    /// file. The output file name will be generated using the provided index (e.g., "mnist_image_0.jpg",
    /// "mnist_image_1.jpg", etc.).
    /// </remarks>
    /// <param name="index">
    /// The zero-based index (0..imageCount) of the image to save. Used to generate the output file name.
    /// </param>
    /// <param name="mnistData">
    /// A two-dimensional array of size (imageCount, 784) containing the pixel values of the MNIST image. Each element
    /// represents a grayscale intensity value from 0 to 255.
    /// </param>
    /// <returns>The full file path of the saved JPG image file.</returns>
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "Display one warning at begining of the method")]
    public static string SaveMnistPicture(int size, int index, float[,] mnistData, string fileName)
    {

#warning The SaveMnistPicture method uses System.Drawing, which may not be fully supported on all platforms. Ensure that the necessary dependencies are available and that the application is run in an environment that supports System.Drawing (e.g., Windows).
        
        float[] imageData = mnistData.GetRow(index);

        // Build the image from the pixel data
        using var source = new System.Drawing.Bitmap(28, 28, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                int i = (y * 28) + x;
                int value = (int)Math.Clamp(imageData[i], 0f, 255f);
                var color = System.Drawing.Color.FromArgb(value, value, value);
                source.SetPixel(x, y, color);
            }
        }

        using var resized = new System.Drawing.Bitmap(size, size, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
        using (var g = System.Drawing.Graphics.FromImage(resized))
        {
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;
            g.DrawImage(source, 0, 0, size, size);
        }

        fileName = $"mnist_image_{fileName}.jpg";
        string filePath = Path.Combine(System.IO.Directory.GetCurrentDirectory(), fileName);
        resized.Save(filePath, System.Drawing.Imaging.ImageFormat.Jpeg);

        return filePath;
    }
}
