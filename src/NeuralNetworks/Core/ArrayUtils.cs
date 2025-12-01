// Machine Learning Utils
// File name: ArrayUtils.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

public class ArrayUtils
{
    public static float[,] LoadCsv(string filePath)
    {
        string[] lines = File.ReadAllLines(filePath);
        int rows = lines.Length;
        int cols = lines[0].Split(',').Length;
        float[,] matrix = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            string[] values = lines[i].Split(',');
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = float.Parse(values[j]);
            }
        }
        return matrix;
    }

    [Conditional("DEBUG")]
    public static void EnsureSameShape(float[,,,]? matrix1, float[,,,]? matrix2)
    {
        if (matrix1 is null || matrix2 is null)
            throw new ArgumentException("Matrix is null.");

        if (!matrix1.HasSameShape(matrix2))
            throw new Exception("Matrices must have the same shape.");
    }

    [Conditional("DEBUG")]
    public static void EnsureSameShape(float[,]? matrix1, float[,]? matrix2)
    {
        if (matrix1 is null || matrix2 is null)
            throw new ArgumentException("Matrix is null.");

        if (!matrix1.HasSameShape(matrix2))
            throw new Exception("Matrices must have the same shape.");
    }

    [Conditional("DEBUG")]
    public static void EnsureSameShape(float[]? matrix1, float[]? matrix2)
    {
        if (matrix1 is null || matrix2 is null)
            throw new ArgumentException("Matrix is null.");

        if (!matrix1.HasSameShape(matrix2))
            throw new Exception("Matrices must have the same shape.");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] CreateRandomNormal(int rows, int columns, Random random, float mean = 0, float stdDev = 1)
    {
        float[,] res = new float[rows, columns];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = BoxMuller() * stdDev + mean;
            }
        }
        return res;

        float BoxMuller() // TODO: Move to RandomUtils
        {
            // uniform(0,1] random doubles
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();

            //random normal(0,1)
            float randStdNormal = Convert.ToSingle(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
            return randStdNormal;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] CreateRandomNormal(int columns, Random random, float mean = 0, float stdDev = 1)
    {
        float[] res = new float[columns];
        for (int col = 0; col < columns; col++)
        {
            res[col] = BoxMuller() * stdDev + mean;
        }
        return res;

        float BoxMuller() // TODO: Move to RandomUtils
        {
            // uniform(0,1] random doubles
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();

            //random normal(0,1)
            float randStdNormal = Convert.ToSingle(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
            return randStdNormal;
        }
    }

    /// <summary>
    /// Creates a new matrix filled with random values between -0.5 and 0.5, with the specified number of rows and columns.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>A new matrix filled with random values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] CreateRandom(int rows, int columns, Random random)
    {
        // Create an instance of Array of floats using rows and columns and fill it with randoms.
        float[,] res = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = random.NextSingle() - 0.5f;
            }
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] CreateRandom(int dim1, int dim2, int dim3, int dim4, Random _random)
    {
        float[,,,] res = new float[dim1, dim2, dim3, dim4];
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        res[i, j, k, l] = _random.NextSingle() - 0.5f;
                    }
                }
            }
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] CreateRandom(int columns, Random random)
    {
        // Create an instance of Array of floats using columns and fill it with randoms.
        float[] res = new float[columns];
        for (int i = 0; i < columns; i++)
        {
            res[i] = random.NextSingle() - 0.5f;
        }
        return res;
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] CreateZeros(int columns) 
        => new float[columns];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] CreateZeros(float[,] matrix) 
        => new float[matrix.GetLength(0), matrix.GetLength(1)];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] CreateZeros(float[,,,] matrix) 
        => new float[matrix.GetLength(0), matrix.GetLength(1), matrix.GetLength(2), matrix.GetLength(3)];

    public static (float[,] xPermuted, float[,] yPermuted) PermuteData(float[,] x, float[,] y, Random random)
    {
        Debug.Assert(x.GetLength((int)Dimension.Rows) == y.GetLength((int)Dimension.Rows));

        int[] indices = [.. Enumerable.Range(0, x.GetLength((int)Dimension.Rows)).OrderBy(i => random.Next())];

        float[,] xPermuted = CreateZeros(x);
        float[,] yPermuted = CreateZeros(y);

        for (int i = 0; i < x.GetLength((int)Dimension.Rows); i++)
        {
            //xPermuted[i] = x[indices[i]];
            //yPermuted[i] = y[indices[i]];
            xPermuted.SetRow(i, x.GetRow(indices[i]));
            yPermuted.SetRow(i, y.GetRow(indices[i]));
        }

        return (xPermuted, yPermuted);
    }

    public static (float[,,,] xPermuted, float[,] yPermuted) PermuteData(float[,,,] x, float[,] y, Random random)
    {
        Debug.Assert(x.GetLength((int)Dimension.Rows) == y.GetLength((int)Dimension.Rows));

        int[] indices = [.. Enumerable.Range(0, x.GetLength((int)Dimension.Rows)).OrderBy(i => random.Next())];

        float[,,,] xPermuted = CreateZeros(x);
        float[,] yPermuted = CreateZeros(y);

        for (int i = 0; i < x.GetLength((int)Dimension.Rows); i++)
        {
            //xPermuted[i] = x[indices[i]];
            //yPermuted[i] = y[indices[i]];
            xPermuted.SetRow(i, x.GetRow(indices[i]));
            yPermuted.SetRow(i, y.GetRow(indices[i]));
        }

        return (xPermuted, yPermuted);
    }
}
