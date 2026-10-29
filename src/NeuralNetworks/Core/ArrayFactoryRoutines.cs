// Neural Networks in C♯
// File name: ArrayFactoryRoutines.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

/// <summary>
/// Provides factory and initialization routines for creating single-precision arrays with range-based values, uniformly
/// distributed random values, and normally distributed random values.
/// </summary>
/// <remarks>Includes overloads for multiple array ranks and an in-place extension routine for filling
/// two-dimensional arrays with reproducible random values from a seed.</remarks>
public static class ArrayFactoryRoutines
{
    #region CreateRange

    // TODO: CreateRange - make them consistent with the step calculation and value assignment across all overloads.

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
    public static float[,,,] CreateRange(int dim0, int dim1, int dim2, int dim3, float from, float to)
    {
        float[,,,] res = new float[dim0, dim1, dim2, dim3];
        float step = (to - from) / (dim0 * dim1 * dim2 * dim3);
        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    for (int l = 0; l < dim3; l++)
                    {
                        res[i, j, k, l] = from + step * (i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + l);
                    }
                }
            }
        }
        return res;
    }

    #endregion

    #region CreateRandom

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] CreateRandomNormal(int rows, int columns, Random random, float mean = 0, float stdDev = 1)
    {
        float[,] res = new float[rows, columns];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = NextBoxMuller(random) * stdDev + mean;
            }
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] CreateRandomNormal(int columns, Random random, float mean = 0, float stdDev = 1)
    {
        float[] res = new float[columns];
        for (int col = 0; col < columns; col++)
        {
            res[col] = NextBoxMuller(random) * stdDev + mean;
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,] CreateRandomNormal(int dim0, int dim1, int dim2, Random random, float mean = 0, float stdDev = 1)
    {
        float[,,] res = new float[dim0, dim1, dim2];
        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    res[i, j, k] = NextBoxMuller(random) * stdDev + mean;
                }
            }
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] CreateRandomNormal(int dim0, int dim1, int dim2, int dim3, Random random, float mean = 0, float stdDev = 1)
    {
        float[,,,] res = new float[dim0, dim1, dim2, dim3];
        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    for (int l = 0; l < dim3; l++)
                    {
                        res[i, j, k, l] = NextBoxMuller(random) * stdDev + mean;
                    }
                }
            }
        }
        return res;
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
    public static float[,,,] CreateRandom(int dim0, int dim1, int dim2, int dim3, Random random)
    {
        float[,,,] res = new float[dim0, dim1, dim2, dim3];
        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    for (int l = 0; l < dim3; l++)
                    {
                        res[i, j, k, l] = random.NextSingle() - 0.5f;
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

    /// <summary>
    /// Fills the source with random float values between -0.5 and 0.5 using the specified seed.
    /// </summary>
    /// <param name="source">The two-dimensional array to fill with random values.</param>
    /// <param name="seed">The seed used to initialize the random number generator.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void RandomInPlace(this float[,] source, int seed)
    {
        Random rand = new(seed);
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                source[row, col] = rand.NextSingle() - 0.5f;
            }
        }
    }

    /// <summary>
    /// Generates a random number following a standard normal distribution (mean = 0, stdDev = 1) using the Box-Muller transform.
    /// </summary>
    /// <param name="random"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float NextBoxMuller(Random random)
    {
        // uniform(0,1] random floats
        float u1 = 1 - random.NextSingle();
        float u2 = 1 - random.NextSingle();

        //random normal(0,1)
        float randStdNormal = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Sin(2.0f * MathF.PI * u2);
        return randStdNormal;
    }

    #endregion
}
