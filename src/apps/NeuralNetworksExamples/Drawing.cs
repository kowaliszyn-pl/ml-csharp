// Neural Networks in C♯
// File name: Drawing.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics.CodeAnalysis;
using System.Drawing;

using NeuralNetworks.Core;

namespace NeuralNetworksExamples;

[SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "Display one warning at beginning of the method")]
public class Drawing
{
#warning The NeuralNetworksExamples.Drawing class uses System.Drawing, which may not be fully supported on all platforms. Ensure that the necessary dependencies are available and that the application is run in an environment that supports System.Drawing (e.g., Windows).

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
    public static string SaveMnistPicture(int size, int index, float[,] mnistData, string fileName)
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

    public static string SaveEcg200Picture(int chartWidth, int chartHeight, int margin, int index, float[,] ecgData, string fileName)
    {
        float[] chartData = ecgData.GetRow(index);

        // Build the ECG chart from the data
        using Bitmap bitmap = new(chartWidth, chartHeight, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

        using Graphics graphics = Graphics.FromImage(bitmap);
        graphics.Clear(Color.White);

        // Define the drawing area (canvas) inside the margins
        int canvasLeft = margin;
        int canvasTop = margin;
        int canvasWidth = chartWidth - (2 * margin);
        int canvasHeight = chartHeight - (2 * margin);

        // Find the min and max values in the data to scale the chart
        float minValue = chartData.Min();
        float maxValue = chartData.Max();

        // Draw some horizontal and vertical grid lines for better visibility
        using Pen gridPen = new(Color.LightGray, 1);
        for (int i = 0; i < 11; i++)
        {
            float y = canvasTop + (canvasHeight * i / 10f);
            graphics.DrawLine(gridPen, canvasLeft, y, canvasLeft + canvasWidth, y);
        }
        for (int i = 0; i < 11; i++)
        {
            float x = canvasLeft + (canvasWidth * i / 10f);
            graphics.DrawLine(gridPen, x, canvasTop, x, canvasTop + canvasHeight);
        }

        // Draw the ECG line
        using Pen redPen = new(Color.Red, 2);

        if(chartData.Length < 2)
        {
            throw new ArgumentException("Chart data must contain at least two points to draw a line.");
        }

        float xStep = canvasWidth / (float)(chartData.Length - 1);
        float valueRange = maxValue - minValue;
        float yScale = valueRange == 0 ? 1 : canvasHeight / valueRange;
        int canvasBottom = canvasTop + canvasHeight;

        // Draw lines between consecutive points
        for (int i = 1; i < chartData.Length; i++)
        {
            float x1 = canvasLeft + ((i - 1) * xStep);
            float y1 = canvasBottom - ((chartData[i - 1] - minValue) * yScale);
            float x2 = canvasLeft + (i * xStep);
            float y2 = canvasBottom - ((chartData[i] - minValue) * yScale);
            graphics.DrawLine(redPen, x1, y1, x2, y2);
        }

        fileName = $"ecg200-chart-{fileName}.jpg";
        string filePath = Path.Combine(Directory.GetCurrentDirectory(), fileName);
        bitmap.Save(filePath, System.Drawing.Imaging.ImageFormat.Jpeg);

        return filePath;
    }

    public static string SaveSineChart(int sineChartWidth, int sineChartHeight, int sineChartMargin, List<(float x, float yActual, float yPredicted)> chartData, int verticalLines, string fileName)
    {
        // Draw 2 lines - one for actual values and one for predicted values - on the same chart, with x values from -2π to 2π and y values scaled to fit the chart height. The chart should have a white background, light gray grid lines, and the actual values line should be blue while the predicted values line should be red.
        using Bitmap bitmap = new(sineChartWidth, sineChartHeight, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

        using Graphics graphics = Graphics.FromImage(bitmap);
        graphics.Clear(Color.White);

        // Define the drawing area (canvas) inside the margins
        int canvasLeft = sineChartMargin;
        int canvasTop = sineChartMargin;
        int canvasWidth = sineChartWidth - (2 * sineChartMargin);
        int canvasHeight = sineChartHeight - (2 * sineChartMargin);

        // Find the min and max values in the data to scale the chart
        float minValue = chartData.Min(p => Math.Min(p.yActual, p.yPredicted));
        float maxValue = chartData.Max(p => Math.Max(p.yActual, p.yPredicted));

        // Draw 'verticalLines' vertical and 11 horizontal grid lines for better visibility
        using Pen gridPen = new(Color.LightGray, 1);
        for (int i = 0; i < 11; i++)
        {
            float y = canvasTop + (canvasHeight * i / 10f);
            graphics.DrawLine(gridPen, canvasLeft, y, canvasLeft + canvasWidth, y);
        }
        for (int i = 0; i < verticalLines; i++)
        {
            float x = canvasLeft + (canvasWidth * i / (verticalLines - 1f));
            graphics.DrawLine(gridPen, x, canvasTop, x, canvasTop + canvasHeight);
        }

        // Draw the chart lines
        using Pen actualPen = new(Color.Blue, 2);
        using Pen predictedPen = new(Color.Red, 2);

        float xStep = canvasWidth / (float)(chartData.Count - 1);
        float yScale = (maxValue - minValue) == 0 ? 1 : canvasHeight / (maxValue - minValue);
        int canvasBottom = canvasTop + canvasHeight;

        // Draw lines between consecutive points for both actual and predicted values

        for (int i = 1; i < chartData.Count; i++)
        {
            float x1 = canvasLeft + ((i - 1) * xStep);
            float yActual1 = canvasBottom - ((chartData[i - 1].yActual - minValue) * yScale);
            float yPredicted1 = canvasBottom - ((chartData[i - 1].yPredicted - minValue) * yScale);
            float x2 = canvasLeft + (i * xStep);
            float yActual2 = canvasBottom - ((chartData[i].yActual - minValue) * yScale);
            float yPredicted2 = canvasBottom - ((chartData[i].yPredicted - minValue) * yScale);
            graphics.DrawLine(actualPen, x1, yActual1, x2, yActual2);
            graphics.DrawLine(predictedPen, x1, yPredicted1, x2, yPredicted2);
        }

        fileName = $"sine-chart-{fileName}.jpg";
        string filePath = Path.Combine(Directory.GetCurrentDirectory(), fileName);
        bitmap.Save(filePath, System.Drawing.Imaging.ImageFormat.Jpeg);

        return filePath;
    }

    #endregion
}
