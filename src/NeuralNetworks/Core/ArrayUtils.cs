// Neural Networks in C♯
// File name: ArrayUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using ILGPU.Runtime.Cuda;

using NeuralNetworks.Core;

namespace NeuralNetworks.Core;

public class ArrayUtils
{
    public static float[,] LoadCsv(string filePath, int skipHeaderLines = 0)
        => LoadSv(filePath, ',', skipHeaderLines);

    public static float[,] LoadTsv(string filePath, int skipHeaderLines = 0)
        => LoadSv(filePath, '\t', skipHeaderLines);

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

    

    public static Span<float> ConvertToSpan(Array array)
    {
        return array switch
        {
            float[] arr1D => arr1D.AsSpan(),
            float[,] arr2D => MemoryMarshal.CreateSpan(ref arr2D[0, 0], arr2D.Length),
            float[,,] arr3D => MemoryMarshal.CreateSpan(ref arr3D[0, 0, 0], arr3D.Length),
            float[,,,] arr4D => MemoryMarshal.CreateSpan(ref arr4D[0, 0, 0, 0], arr4D.Length),
            _ => throw new ArgumentException("Array must be of type float[] or float[,] or float[,,,].")
        };

        //if (array is float[] arr1D)
        //    return arr1D.AsSpan();
        //else if (array is float[,] arr2D)
        //    return MemoryMarshal.CreateSpan(ref arr2D[0, 0], arr2D.Length);
        //else if (array is float[,,,] arr4D)
        //    return MemoryMarshal.CreateSpan(ref arr4D[0, 0, 0, 0], arr4D.Length);
        //else
        //    throw new ArgumentException("Array must be of type float[] or float[,] or float[,,,].");
    }

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

    public static void StandardizeColumns(float minStdDev, params List<float[,]> sets)
    {
        // Assert all sets have the same number of columns
        int columns = sets[0].GetLength(1);
#if DEBUG
        foreach (var set in sets)
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
