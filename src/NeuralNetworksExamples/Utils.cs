// Neural Networks in C♯
// File name: Utils.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

using static System.Console;

namespace NeuralNetworksExamples;

internal static class Utils
{
    public static void DisplayDigit3PredictionExamples(float[,] yTest, float[,] logits, float[,] xTest)
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

        SaveMnistPicture(200, correctlyPredicted3, xTest);
        SaveMnistPicture(200, correctlyPredictedNot3, xTest);
        SaveMnistPicture(200, incorrectlyPredicted3, xTest);
        SaveMnistPicture(200, incorrectlyPredictedNot3, xTest);

        // Print the results
        WriteLine("Examples of predictions vs actual values for the digit \"3\":");
        WriteLine($"1. \"3\" that was correctly predicted as \"3\": index {correctlyPredicted3}");
        WriteLine($"2. Not \"3\" that was correctly predicted as not \"3\": index {correctlyPredictedNot3}");
        WriteLine($"3. \"3\" that was incorrectly predicted as not \"3\": index {incorrectlyPredicted3}");
        WriteLine($"4. Not \"3\" that was incorrectly predicted as \"3\": index {incorrectlyPredictedNot3}");
        WriteLine("The corresponding images have been saved as JPG files in the current directory with names based on their indexes (e.g., \"mnist_image_0.jpg\", \"mnist_image_1.jpg\", etc.).");
        WriteLine();
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
    public static string SaveMnistPicture(int size, int index, float[,] mnistData)
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

        string fileName = $"mnist_image_{index}.jpg";
        string filePath = Path.Combine(System.IO.Directory.GetCurrentDirectory(), fileName);
        resized.Save(filePath, System.Drawing.Imaging.ImageFormat.Jpeg);

        return filePath;
    }
}
