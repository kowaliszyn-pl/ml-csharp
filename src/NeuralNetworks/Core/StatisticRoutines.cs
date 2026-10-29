// Neural Networks in C♯
// File name: StatisticRoutines.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

/// <summary>
/// Provides extension methods for computing common statistics on single- and multi-dimensional <see
/// cref="System.Single"/> arrays, including argmax, min, max, sum, mean, standard deviation, and per-column aggregates.
/// </summary>
/// <remarks>Methods are implemented for one-dimensional vectors and for two-, three-, and four-dimensional
/// arrays. Standard deviation is computed as population standard deviation (dividing by the total element
/// count).</remarks>
public static class StatisticRoutines
{
    /// <summary>
    /// Computes the index of the maximum value for each row.
    /// </summary>
    /// <param name="source">The array to evaluate.</param>
    /// <returns>The index of the maximum value in the array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Argmax(this float[] source)
    {
        Debug.Assert(source.Length > 0, "Source array must contain at least one element to compute argmax.");

        int items = source.Length;

        float max = float.MinValue;
        int maxIndex = 0;
        for (int i = 0; i < items; i++)
        {
            float value = source[i];
            if (value > max)
            {
                max = value;
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /// <summary>
    /// Computes the index of the maximum value for each row.
    /// </summary>
    /// <param name="source">The two-dimensional array to evaluate.</param>
    /// <returns>
    /// An array of length equal to the number of rows, where each element is the column index of the maximum value in
    /// that row.
    /// </returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int[] Argmax(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        int[] array = new int[rows];

        for (int row = 0; row < rows; row++)
        {
            float max = float.MinValue;
            int maxIndex = 0;
            for (int col = 0; col < columns; col++)
            {
                float value = source[row, col];
                if (value > max)
                {
                    max = value;
                    maxIndex = col;
                }
            }
            array[row] = maxIndex;
        }

        return array;
    }

    /// <summary>
    /// Returns the maximum element value across all elements of a two-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The maximum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(this float[,] source)
    {
        float max = float.MinValue;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                max = Math.Max(max, source[row, col]);
            }
        }
        return max;
    }

    /// <summary>
    /// Returns the maximum element value across all elements of a three-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The maximum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(this float[,,] source)
    {
        float max = float.MinValue;

        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    max = Math.Max(max, source[i, j, k]);
                }
            }
        }

        return max;
    }

    /// <summary>
    /// Returns the maximum element value across all elements of a four-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The maximum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(this float[,,,] source)
    {
        float max = float.MinValue;

        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);
        int dim3 = source.GetLength(3);

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    for (int l = 0; l < dim3; l++)
                    {
                        max = Math.Max(max, source[i, j, k, l]);
                    }
                }
            }
        }

        return max;
    }

    /// <summary>
    /// Calculates the mean of all elements in the source.
    /// </summary>
    /// <param name="source">The array whose mean will be computed.</param>
    /// <returns>The arithmetic mean of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Mean(this float[,] source)
        => source.Sum() / source.Length;

    /// <summary>
    /// Calculates the mean of all elements in the source.
    /// </summary>
    /// <param name="source">The array whose mean will be computed.</param>
    /// <returns>The arithmetic mean of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Mean(this float[,,] source)
        => source.Sum() / source.Length;

    /// <summary>
    /// Returns the minimum element value across all elements of a two-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The minimum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(this float[,] source)
    {
        float min = float.MaxValue;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                min = Math.Min(min, source[row, col]);
            }
        }
        return min;
    }

    /// <summary>
    /// Returns the minimum element value across all elements of a three-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The minimum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(this float[,,] source)
    {
        float min = float.MaxValue;

        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    min = Math.Min(min, source[i, j, k]);
                }
            }
        }

        return min;
    }

    /// <summary>
    /// Returns the minimum element value across all elements of a four-dimensional array.
    /// </summary>
    /// <param name="source">The array to scan.</param>
    /// <returns>The minimum value found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(this float[,,,] source)
    {
        float min = float.MaxValue;

        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);
        int dim3 = source.GetLength(3);

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    for (int l = 0; l < dim3; l++)
                    {
                        min = Math.Min(min, source[i, j, k, l]);
                    }
                }
            }
        }

        return min;
    }

    /// <summary>
    /// Calculates the mean of all elements in the source.
    /// </summary>
    /// <param name="source">The array whose mean will be computed.</param>
    /// <returns>The arithmetic mean of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Mean(this float[,,,] source)
        => source.Sum() / source.Length;

    /// <summary>
    /// Calculates the mean of each column in the source.
    /// </summary>
    /// <param name="source">The two-dimensional array to process.</param>
    /// <returns>A one-dimensional array where each element is the mean of a column.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] MeanByColumn(this float[,] source)
    {
        int rows = source.GetLength(0);

        Debug.Assert(rows > 0, "Source array must contain at least one row to compute column means.");

        int columns = source.GetLength(1);

        float[] res = new float[columns];

        for (int col = 0; col < columns; col++)
        {
            float sum = 0;
            for (int row = 0; row < rows; row++)
            {
                sum += source[row, col];
            }
            res[col] = sum / rows;
        }

        return res;
    }

    /// <summary>
    /// Calculates the standard deviation for all elements of a two-dimensional array.
    /// </summary>
    /// <remarks>
    /// Standard deviation is calculated using the formula: <c> sqrt(sum((x - mean)^2) / N)</c>, where N is the number
    /// of all elements in the array.
    /// </remarks>
    /// <param name="source">The array whose standard deviation will be computed.</param>
    /// <returns>The standard deviation of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float StdDev(this float[,] source)
    {
        Debug.Assert(source.Length > 0, "Source array must contain at least one element to compute standard deviation.");

        float mean = source.Mean();
        float sum = 0;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float value = source[i, j] - mean;
                sum += value * value;
            }
        }

        return MathF.Sqrt(sum / source.Length);
    }

    /// <summary>
    /// Calculates the standard deviation for all elements of a three-dimensional array.
    /// </summary>
    /// <remarks>
    /// Standard deviation is calculated using the formula: <c> sqrt(sum((x - mean)^2) / N)</c>, where N is the number
    /// of all elements in the array.
    /// </remarks>
    /// <param name="source">The array whose standard deviation will be computed.</param>
    /// <returns>The standard deviation of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float StdDev(this float[,,] source)
    {
        Debug.Assert(source.Length > 0, "Source array must contain at least one element to compute standard deviation.");

        float mean = source.Mean();
        float sum = 0;

        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    float value = source[i, j, k] - mean;
                    sum += value * value;
                }
            }
        }

        return MathF.Sqrt(sum / source.Length);
    }

    /// <summary>
    /// Calculates the standard deviation for all elements of a four-dimensional array.
    /// </summary>
    /// <remarks>
    /// Standard deviation is calculated using the formula: <c> sqrt(sum((x - mean)^2) / N)</c>, where N is the number
    /// of all elements in the array.
    /// </remarks>
    /// <param name="source">The array whose standard deviation will be computed.</param>
    /// <returns>The standard deviation of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float StdDev(this float[,,,] source)
    {
        Debug.Assert(source.Length > 0, "Source array must contain at least one element to compute standard deviation.");

        float mean = source.Mean();
        float sum = 0;

        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);
        int dim3 = source.GetLength(3);

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    for (int l = 0; l < dim3; l++)
                    {
                        float value = source[i, j, k, l] - mean;
                        sum += value * value;
                    }
                }
            }
        }

        return MathF.Sqrt(sum / source.Length);
    }

    /// <summary>
    /// Calculates the sum of all elements in the two-dimensional array.
    /// </summary>
    /// <param name="source">The array to sum.</param>
    /// <returns>The sum of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(this float[,] source)
    {
        // Sum over all elements.
        float sum = 0;
        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                sum += source[row, col];
            }
        }

        return sum;
    }

    /// <summary>
    /// Calculates the sum of all elements in the three-dimensional array.
    /// </summary>
    /// <param name="source">The array to sum.</param>
    /// <returns>The sum of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(this float[,,] source)
    {
        // Sum over all elements.
        float sum = 0;

        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    sum += source[i, j, k];
                }
            }
        }

        return sum;
    }

    /// <summary>
    /// Calculates the sum of all elements in the four-dimensional array.
    /// </summary>
    /// <param name="source">The array to sum.</param>
    /// <returns>The sum of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(this float[,,,] source)
    {
        // Sum over all elements.
        float sum = 0;

        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);
        int dim3 = source.GetLength(3);

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    for (int l = 0; l < dim3; l++)
                    {
                        sum += source[i, j, k, l];
                    }
                }
            }
        }

        return sum;
    }

    /// <summary>
    /// Calculates the sum of each column in the source.
    /// </summary>
    /// <param name="source">The two-dimensional array to process.</param>
    /// <returns>A one-dimensional array where each element is the sum of a column.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] SumByColumns(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[] res = new float[columns];

        for (int col = 0; col < columns; col++)
        {
            float sum = 0;
            for (int row = 0; row < rows; row++)
            {
                sum += source[row, col];
            }
            res[col] = sum;
        }

        return res;
    }
}
