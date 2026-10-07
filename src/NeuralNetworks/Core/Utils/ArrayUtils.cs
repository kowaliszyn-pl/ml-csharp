// Neural Networks in C♯
// File name: ArrayUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Runtime.CompilerServices;

using NeuralNetworks.Core;

namespace NeuralNetworks.Core.Utils;

public class ArrayUtils
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] CreateRange(int rows, int columns, float from, float to)
    {
        float[,] res = new float[rows, columns];
        // float step = (to - from) / (rows * columns);
        float step = (to - from) / columns;
        for (int i = 0; i < rows; i++)
        {
            float value = from;
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = value;
                value += step;
            }
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] CreateRange(int dim1, int dim2, int dim3, int dim4, float from, float to)
    {
        float[,,,] res = new float[dim1, dim2, dim3, dim4];
        float step = (to - from) / (dim1 * dim2 * dim3 * dim4);
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        res[i, j, k, l] = from + step * (i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l);
                    }
                }
            }
        }
        return res;
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
            variance[col] = (sumOfSquares / rows) - (mean[col] * mean[col]);
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
}
