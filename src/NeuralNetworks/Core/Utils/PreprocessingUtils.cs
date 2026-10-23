// Neural Networks in C♯
// File name: PreprocessingUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Utils;

/// <summary>
/// Provides utility methods for loading, reshaping, scaling, splitting, and standardizing numeric datasets represented
/// as multidimensional <see langword="float"/> arrays.
/// </summary>
/// <remarks>Includes both in-place and copy-based operations, with extension methods targeting 2D and 4D arrays
/// for common data-preprocessing workflows.</remarks>
public static class PreprocessingUtils
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

    public static void StandardizeColumns(float minStdDev, params List<float[,]> sets)
    {
        // Assert all sets have the same number of columns
        int columns = sets[0].GetLength(1);
#if DEBUG
        foreach (float[,] set in sets)
        {
            if (set.GetLength(1) != columns)
                throw new ArgumentException("All sets must have the same number of columns.");
        }
#endif

        int rows = sets.Sum(s => s.GetLength(0));

        // Compute mean and stdDev for each column using both train and test
        float[] mean = new float[columns];
        float[] stdDev = new float[columns];
        float[] variance = new float[columns];

        for (int col = 0; col < columns; col++)
        {
            float sum = 0f;
            float sumOfSquares = 0f;

            // Calculate sum and sum of squares for the current column across all sets
            foreach (float[,] set in sets)
            {
                int rowsInSet = set.GetLength(0);
                for (int row = 0; row < rowsInSet; row++)
                {
                    float val = set[row, col];
                    sum += val;
                    sumOfSquares += val * val;
                }
            }

            mean[col] = sum / rows;
            variance[col] = MathF.Max((sumOfSquares / rows) - (mean[col] * mean[col]), 0f); // Ensure variance is non-negative
            stdDev[col] = MathF.Max(MathF.Sqrt(variance[col]), minStdDev);
            if (stdDev[col] == 0f)
            {
                stdDev[col] = 1f; // Prevent division by zero
            }
        }

        // Update each set
        foreach (float[,] set in sets)
        {
            int rowsInSet = set.GetLength(0);
            // Standardize set
            for (int row = 0; row < rowsInSet; row++)
                for (int col = 0; col < columns; col++)
                    set[row, col] = (set[row, col] - mean[col]) / stdDev[col];
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

    /// <summary>
    /// Standardizes the source in-place so that each column (or a specified range of dim2) has a mean of 0 and a
    /// standard deviation of 1 by applying the <c> (x - mean) / stdDev</c> transformation.
    /// </summary>
    /// <remarks>
    /// Standard deviation is calculated using the formula: <c> sqrt(sum((x - mean)^2) / N)</c>, where N is the number
    /// of dim1.
    /// </remarks>
    /// <param name="source">The source to standardize.</param>
    /// <param name="columnRange">
    /// Optional. The range of dim2 to standardize. If null, all dim2 are standardized.
    /// </param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StandardizeByColumns(this float[,] source, Range? columnRange = null)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        int beginColumn, endColumn;

        if (columnRange is not null)
        {
            (int offset, int length) = columnRange.Value.GetOffsetAndLength(columns);
            beginColumn = offset;
            endColumn = beginColumn + length;
        }
        else
        {
            beginColumn = 0;
            endColumn = columns;
        }

        for (int col = beginColumn; col < endColumn; col++)
        {
            // Calculate mean
            float sum = 0;
            for (int row = 0; row < rows; row++)
            {
                sum += source[row, col];
            }
            float mean = sum / rows;

            // Calculate standard deviation
            float sumOfSquares = 0;
            for (int row = 0; row < rows; row++)
            {
                float value = source[row, col] - mean;
                sumOfSquares += value * value;
            }
            float stdDev = MathF.Sqrt(sumOfSquares / rows);

            if (stdDev == 0)
            {
                stdDev = 1; // To avoid division by zero
            }

            // Standardize values
            for (int row = 0; row < rows; row++)
            {
                source[row, col] = (source[row, col] - mean) / stdDev;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StandardizeByRowsInPlace(this float[,] source, Range? rowRange = null)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        int beginRow, endRow;
        if (rowRange is not null)
        {
            (int offset, int length) = rowRange.Value.GetOffsetAndLength(rows);
            beginRow = offset;
            endRow = beginRow + length;
        }
        else
        {
            beginRow = 0;
            endRow = rows;
        }

        for (int row = beginRow; row < endRow; row++)
        {
            // Calculate mean
            float sum = 0;
            for (int col = 0; col < columns; col++)
            {
                sum += source[row, col];
            }
            float mean = sum / columns;

            // Calculate standard deviation
            float sumOfSquares = 0;
            for (int col = 0; col < columns; col++)
            {
                float value = source[row, col] - mean;
                sumOfSquares += value * value;
            }
            float stdDev = MathF.Sqrt(sumOfSquares / columns);
            if (stdDev == 0)
            {
                stdDev = 1; // To avoid division by zero
            }

            // Standardize values
            for (int col = 0; col < columns; col++)
            {
                source[row, col] = (source[row, col] - mean) / stdDev;
            }
        }

    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] StandardizeByRows(this float[,] source, Range? rowRange = null)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] result = new float[rows, columns];

        int beginRow, endRow;
        if (rowRange is not null)
        {
            (int offset, int length) = rowRange.Value.GetOffsetAndLength(rows);
            beginRow = offset;
            endRow = beginRow + length;
        }
        else
        {
            beginRow = 0;
            endRow = rows;
        }

        for (int row = beginRow; row < endRow; row++)
        {
            // Calculate mean
            float sum = 0;
            for (int col = 0; col < columns; col++)
            {
                sum += source[row, col];
            }
            float mean = sum / columns;

            // Calculate standard deviation
            float sumOfSquares = 0;
            for (int col = 0; col < columns; col++)
            {
                float value = source[row, col] - mean;
                sumOfSquares += value * value;
            }
            float stdDev = MathF.Sqrt(sumOfSquares / columns);
            if (stdDev == 0)
            {
                stdDev = 1; // To avoid division by zero
            }

            // Standardize values
            for (int col = 0; col < columns; col++)
            {
                result[row, col] = (source[row, col] - mean) / stdDev;
            }
        }
        return result;

    }

    /// <summary>
    /// Standardizes the source in-place so that each column (or a specified range of dim2) has mean 0 and standard
    /// deviation 1, computed in a single pass.
    /// </summary>
    /// <param name="source">The source to standardize.</param>
    /// <param name="columnRange">
    /// Optional. The range of dim2 to standardize. If null, all dim2 are standardized.
    /// </param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StandardizeSinglePass(this float[,] source, Range? columnRange = null)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        int beginColumn, endColumn;

        if (columnRange is not null)
        {
            (int offset, int length) = columnRange.Value.GetOffsetAndLength(columns);
            beginColumn = offset;
            endColumn = beginColumn + length;
        }
        else
        {
            beginColumn = 0;
            endColumn = columns;
        }

        for (int col = beginColumn; col < endColumn; col++)
        {
            // Calculate standard deviation in a single pass
            float sum = 0, sumOfSquares = 0;
            for (int row = 0; row < rows; row++)
            {
                float value = source[row, col];
                sum += value;
                sumOfSquares += value * value;
            }
            float mean = sum / rows;
            float variance = MathF.Max((sumOfSquares / rows) - (mean * mean), 0f); // Ensure variance is non-negative
            float stdDev = MathF.Sqrt(variance);

            if (stdDev == 0)
            {
                stdDev = 1; // To avoid division by zero
            }

            // Standardize values
            for (int row = 0; row < rows; row++)
            {
                source[row, col] = (source[row, col] - mean) / stdDev;
            }
        }
    }
}
