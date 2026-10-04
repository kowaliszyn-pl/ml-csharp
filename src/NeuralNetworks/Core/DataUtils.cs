// Neural Networks in C♯
// File name: DataUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

public static class DataUtils
{
    /// <summary>
    /// Standardize the 2D array in place by subtracting the given mean and dividing by the given standard deviation.
    /// </summary>
    /// <param name="data"></param>
    /// <param name="mean"></param>
    /// <param name="stdDev"></param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplyStandardizationInPlace(this float[,] data, float mean, float stdDev)
    {
        data.AddInPlace(-mean);
        data.DivideInPlace(stdDev);
    }

    /// <summary>
    /// Apply standardization to the 4D array in place by subtracting the given mean and dividing by the given standard deviation.
    /// </summary>
    /// <param name="data"></param>
    /// <param name="mean"></param>
    /// <param name="stdDev"></param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplyStandardizationInPlace(this float[,,,] data, float mean, float stdDev)
    {
        data.AddInPlace(-mean);
        data.DivideInPlace(stdDev);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] LoadCsv(string filePath, int skipHeaderLines = 0)
        => LoadSv(filePath, ',', skipHeaderLines);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] LoadTsv(string filePath, int skipHeaderLines = 0)
        => LoadSv(filePath, '\t', skipHeaderLines);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] LoadSv(string filePath, char separator, int skipHeaderLines)
    {
        string[] lines = [.. File.ReadAllLines(filePath).Skip(skipHeaderLines)];
        int rows = lines.Length;
        int cols = lines[0].Split(separator).Length;
        float[,] matrix = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            string[] values = lines[i].Split(separator);
            for (int j = 0; j < cols; j++)
            {
                string value = values[j].Trim('"');
                matrix[i, j] = float.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
            }
        }
        return matrix;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float[,] xPermuted, float[,] yPermuted) PermuteData(float[,] x, float[,] y, Random random)
    {
        Debug.Assert(x.GetLength(0) == y.GetLength(0));

        int[] indices = [.. Enumerable.Range(0, x.GetLength(0)).OrderBy(i => random.Next())];

        float[,] xPermuted = x.AsZeros();
        float[,] yPermuted = y.AsZeros();

        for (int i = 0; i < x.GetLength(0); i++)
        {
            //xPermuted[i] = x[indices[i]];
            //yPermuted[i] = y[indices[i]];
            xPermuted.SetRow(i, x.GetRow(indices[i]));
            yPermuted.SetRow(i, y.GetRow(indices[i]));
        }

        return (xPermuted, yPermuted);
    }


    /// <summary>
    /// Permutes the data in the input arrays x and y using the provided random number generator.
    /// </summary>
    /// <remarks>
    /// This method is the quickest way to permute data for 4D input arrays.
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="random"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float[,,,] xPermuted, float[,] yPermuted) PermuteData(float[,,,] x, float[,] y, Random random)
    {
        Debug.Assert(x.GetLength(0) == y.GetLength(0));

        int[] indices = [.. Enumerable.Range(0, x.GetLength(0)).OrderBy(i => random.Next())];

        float[,,,] xPermuted = x.AsZeros();
        float[,] yPermuted = y.AsZeros();

        for (int i = 0; i < x.GetLength(0); i++)
        {
            //xPermuted[i] = x[indices[i]];
            //yPermuted[i] = y[indices[i]];
            xPermuted.SetRow(i, x.GetRow(indices[i]));
            yPermuted.SetRow(i, y.GetRow(indices[i]));
        }

        return (xPermuted, yPermuted);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] ReshapeTo2D(this float[,,,] data)
    {
        int rows = data.GetLength(0);
        int channels = data.GetLength(1);
        int height = data.GetLength(2);
        int width = data.GetLength(3);

        float[,] res = new float[rows, channels * height * width];
        for (int i = 0; i < rows; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                int channelOffset = c * height * width;
                for (int h = 0; h < height; h++)
                {
                    int channelHeightOffset = channelOffset + h * width;
                    for (int w = 0; w < width; w++)
                    {
                        res[i, channelHeightOffset + w] = data[i, c, h, w];
                    }
                }
            }
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] ReshapeTo4D(this float[,] data, int channels, int height, int width)
    {
        int rows = data.GetLength(0);
        int columns = data.GetLength(1);

        Debug.Assert(columns == channels * height * width);

        float[,,,] res = new float[rows, channels, height, width];
        for (int i = 0; i < rows; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                int channelOffset = c * height * width;
                for (int h = 0; h < height; h++)
                {
                    int channelHeightOffset = channelOffset + h * width;
                    for (int w = 0; w < width; w++)
                    {
                        res[i, c, h, w] = data[i, channelHeightOffset + w];
                    }
                }
            }
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] ScaleTo(this float[,] data, float newMin, float newMax)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);

        float[,] res = new float[rows, cols];
        ScaleToInternal(data, res, rows, cols, newMin, newMax);
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ScaleToInPlace(this float[,] data, float newMin, float newMax)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);

        ScaleToInternal(data, data, rows, cols, newMin, newMax);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScaleToInternal(float[,] source, float[,] dest, int rows, int cols, float newMin, float newMax)
    {
        float min = float.MaxValue;
        float max = float.MinValue;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float value = source[i, j];

                if (value < min)
                    min = value;

                if (value > max)
                    max = value;
            }
        }

        float range = max - min;
        float scale = range == 0f ? 0f : (newMax - newMin) / range;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                dest[i, j] = ((source[i, j] - min) * scale) + newMin;
            }
        }
    }

    /// <summary>
    /// Standardize the 2D array in place by subtracting the mean and dividing by the standard deviation, and return the mean and standard deviation used for standardization.
    /// </summary>
    /// <param name="data"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float Mean, float StdDev) StandardizeInPlace(this float[,] data)
    {
        float mean = data.Mean();
        data.AddInPlace(-mean);

        float stdDev = data.StdDev();
        data.DivideInPlace(stdDev);

        return (mean, stdDev);
    }

    /// <summary>
    /// Standardize the 4D array in place by subtracting the mean and dividing by the standard deviation, and return the mean and standard deviation used for standardization.
    /// </summary>
    /// <param name="data"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float Mean, float StdDev) StandardizeInPlace(this float[,,,] data)
    {
        float mean = data.Mean();
        data.AddInPlace(-mean);

        float stdDev = data.StdDev();
        data.DivideInPlace(stdDev);

        return (mean, stdDev);
    }
}
