// Neural Networks in C♯
// File name: RandomUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

public class RandomUtils
{
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
    public static float[,,,] CreateRandomNormal(int dim1, int dim2, int dim3, int dim4, Random random, float mean = 0, float stdDev = 1)
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
                        res[i, j, k, l] = NextBoxMuller(random) * stdDev + mean;
                    }
                }
            }
        }
        return res;
    }

    /// <summary>
    /// Generates a random number following a standard normal distribution (mean = 0, stdDev = 1) using the Box-Muller transform.
    /// </summary>
    /// <param name="random"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float NextBoxMuller(Random random)
    {
        // uniform(0,1] random doubles
        double u1 = 1 - random.NextDouble();
        double u2 = 1 - random.NextDouble();

        //random normal(0,1)
        float randStdNormal = Convert.ToSingle(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
        return randStdNormal;
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

}
