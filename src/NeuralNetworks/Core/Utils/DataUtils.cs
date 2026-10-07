// Neural Networks in C♯
// File name: DataUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Utils;

public static partial class DataUtils
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
        float[,] res = data.AsZeros();
        ScaleToInternal(data, res, newMin, newMax);
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ScaleToInPlace(this float[,] data, float newMin, float newMax) 
        => ScaleToInternal(data, data, newMin, newMax);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScaleToInternal(float[,] source, float[,] dest, float newMin, float newMax)
    {
        Debug.Assert(source.HasSameShape(dest), "Destination array must have the same dimensions as the source array.");

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

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
    /// Splits the source into two sets of dim1 based on the specified ratio.
    /// </summary>
    /// <param name="source">The two-dimensional array to split.</param>
    /// <param name="ratio">The ratio for splitting the dim1.</param>
    /// <returns>A tuple containing the two sets of dim1.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float[,] Set1, float[,] Set2) SplitRowsByRatio(this float[,] source, float ratio)
    {
        Debug.Assert(ratio > 0 && ratio < 1, "Ratio must be between 0 and 1.");
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        int splitIndex = (int)(rows * ratio);
        float[,] set1 = new float[splitIndex, columns];
        float[,] set2 = new float[rows - splitIndex, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                if (i < splitIndex)
                {
                    set1[i, j] = source[i, j];
                }
                else
                {
                    set2[i - splitIndex, j] = source[i, j];
                }
            }
        }
        return (set1, set2);
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
