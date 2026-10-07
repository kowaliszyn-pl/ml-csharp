// Neural Networks in C♯
// File name: DataUtils.Permutations.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

public static partial class DataUtils
{
    
    /*
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
    /// Permutes the data in the input arrays x and y using the provided random number generator. It does not use the Fisher-Yates shuffle, but instead creates a random permutation of indices and applies it to both arrays. This method is efficient for permuting 4D input arrays and their corresponding labels, ensuring that the relationship between x and y is preserved after permutation. The method returns new permuted arrays without modifying the original inputs.
    /// algorithm.
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

    /// <summary>
    /// Randomly permutes the dim1 of the source in-place using the specified seed. It uses the Fisher-Yates shuffle
    /// algorithm.
    /// </summary>
    /// <remarks>
    /// Complexity: O(n * m), where n = dim1, m = dim2.
    /// </remarks>
    /// <param name="source">The two-dimensional array whose dim1 will be permuted.</param>
    /// <param name="seed">The seed used to initialize the random number generator.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlace(this float[,] source, int seed)
    {
        Random rand = new(seed);
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        for (int i = rows - 1; i > 0; i--)
        {
            int j = rand.Next(i + 1);
            if (i != j)
            {
                // Swap row i with row j
                for (int col = 0; col < columns; col++)
                {
                    (source[j, col], source[i, col]) = (source[i, col], source[j, col]);
                }
            }
        }
    }

    /// <summary>
    /// Randomly permutes the dim1 of the source in-place using the provided random instance (Fisher-Yates shuffle).
    /// </summary>
    /// <param name="source">The two-dimensional array whose dim1 will be permuted.</param>
    /// <param name="random">The random number generator. If null, a new instance is created.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlace(this float[,] source, Random? random)
    {
        random ??= new();
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        for (int i = rows - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            if (i != j)
            {
                // Swap row i with row j
                for (int col = 0; col < columns; col++)
                {
                    (source[j, col], source[i, col]) = (source[i, col], source[j, col]);
                }
            }
        }
    }

    /// <summary>
    /// Randomly permutes the dim1 of the specified matrices in place, ensuring that corresponding dim1 in both matrices
    /// remain aligned.
    /// </summary>
    /// <remarks>
    /// This method performs an in-place permutation of the dim1 of both matrices, maintaining the correspondence
    /// between dim1. This is useful when shuffling paired data, such as features and labels, for machine learning
    /// tasks. The operation modifies the input matrices directly. <para/> This method is the quickest for permuting two
    /// 2D matrices together.
    /// </remarks>
    /// <param name="source">
    /// The first matrix whose dim1 will be permuted. Must have the same number of dim1 as
    /// <paramref name="secondMatrix"/>.
    /// </param>
    /// <param name="secondMatrix">
    /// The second matrix whose dim1 will be permuted in tandem with <paramref name="source"/>. Must have the same
    /// number of dim1 as <paramref name="source"/>.
    /// </param>
    /// <param name="random">
    /// The random number generator used to determine the permutation order. If <see langword="null"/>, a new instance
    /// will be created.
    /// </param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this float[,] source, float[,] secondMatrix, Random? random)
    {
        random ??= new();
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        int secondColumns = secondMatrix.GetLength(1);

        Debug.Assert(rows == secondMatrix.GetLength(0), "Both matrices must have the same number of dim1 to permute them together.");

        for (int i = rows - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            if (i != j)
            {
                // Swap row i with row j
                for (int col = 0; col < columns; col++)
                {
                    (source[j, col], source[i, col]) = (source[i, col], source[j, col]);
                }

                for (int col = 0; col < secondColumns; col++)
                {
                    (secondMatrix[j, col], secondMatrix[i, col]) = (secondMatrix[i, col], secondMatrix[j, col]);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this int[,] source, float[,] secondMatrix, Random? random)
    {
        random ??= new();
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        int secondColumns = secondMatrix.GetLength(1);

        Debug.Assert(rows == secondMatrix.GetLength(0), "Both matrices must have the same number of dim1 to permute them together.");

        for (int i = rows - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            if (i != j)
            {
                // Swap row i with row j
                for (int col = 0; col < columns; col++)
                {
                    (source[j, col], source[i, col]) = (source[i, col], source[j, col]);
                }

                for (int col = 0; col < secondColumns; col++)
                {
                    (secondMatrix[j, col], secondMatrix[i, col]) = (secondMatrix[i, col], secondMatrix[j, col]);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this float[,,] source, float[,] secondMatrix, Random? random)
    {
        random ??= new();
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        int secondColumns = secondMatrix.GetLength(1);

        Debug.Assert(dim1 == secondMatrix.GetLength(0), "Both matrices must have the same number of dim1 to permute them together.");

        for (int i = dim1 - 1; i > 0; i--)
        {
            int i2 = random.Next(i + 1);
            if (i != i2)
            {
                // Swap row i with row j
                for (int j = 0; j < dim2; j++)
                {
                    for (int k = 0; k < dim3; k++)
                    {
                        (source[i2, j, k], source[i, j, k]) = (source[i, j, k], source[i2, j, k]);
                    }
                }
                for (int col = 0; col < secondColumns; col++)
                {
                    (secondMatrix[i2, col], secondMatrix[i, col]) = (secondMatrix[i, col], secondMatrix[i2, col]);
                }
            }
        }
    }

    /// <summary>
    /// Randomly permutes the dim1 of the specified four-dimensional array and the corresponding dim1 of the second
    /// matrix in place, ensuring that both arrays are shuffled together using the same permutation.
    /// </summary>
    /// <remarks>
    /// Both <paramref name="source"/> and <paramref name="secondMatrix"/> must have the same number of dim1; otherwise,
    /// the method will not perform a valid permutation. The permutation is performed in place and affects the original
    /// arrays. This method is useful for maintaining alignment between related datasets when shuffling.
    /// </remarks>
    /// <param name="source">
    /// The four-dimensional array whose dim1 will be permuted in place. The first dimension represents the dim1 to be
    /// shuffled.
    /// </param>
    /// <param name="secondMatrix">
    /// The two-dimensional matrix whose dim1 will be permuted in place together with the dim1 of
    /// <paramref name="source"/>. Must have the same number of dim1 as <paramref name="source"/>.
    /// </param>
    /// <param name="random">
    /// The random number generator used to determine the permutation order. If <see langword="null"/>, a new instance
    /// of <see cref="Random"/> will be created.
    /// </param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this float[,,,] source, float[,] secondMatrix, Random? random)
    {
        random ??= new();
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);

        int secondColumns = secondMatrix.GetLength(1);

        Debug.Assert(dim1 == secondMatrix.GetLength(0), "Both matrices must have the same number of dim1 to permute them together.");

        for (int i = dim1 - 1; i > 0; i--)
        {
            int i2 = random.Next(i + 1);
            if (i != i2)
            {
                // Swap row i with row j
                for (int j = 0; j < dim2; j++)
                {
                    for (int k = 0; k < source.GetLength(2); k++)
                    {
                        for (int l = 0; l < source.GetLength(3); l++)
                        {
                            (source[i2, j, k, l], source[i, j, k, l]) = (source[i, j, k, l], source[i2, j, k, l]);
                        }
                    }
                }
                for (int col = 0; col < secondColumns; col++)
                {
                    (secondMatrix[i2, col], secondMatrix[i, col]) = (secondMatrix[i, col], secondMatrix[i2, col]);
                }
            }
        }
    }

    /// <summary>
    /// Randomly permutes the rows of the specified four-dimensional array and the corresponding rows of the second
    /// matrix in place, ensuring that both arrays are shuffled together using the same permutation.
    /// </summary>
    /// <remarks>
    /// Both <paramref name="source"/> and <paramref name="secondMatrix"/> must have the same number of dim1; otherwise,
    /// the method will not perform a valid permutation. The permutation is performed in place and affects the original
    /// arrays. This method is useful for maintaining alignment between related datasets when shuffling.
    /// </remarks>
    /// <param name="source">
    /// The four-dimensional array whose dim1 will be permuted in place. The first dimension represents the dim1 to be
    /// shuffled.
    /// </param>
    /// <param name="secondMatrix">
    /// The two-dimensional matrix whose dim1 will be permuted in place together with the dim1 of
    /// <paramref name="source"/>. Must have the same number of dim1 as <paramref name="source"/>.
    /// </param>
    /// <param name="random">
    /// The random number generator used to determine the permutation order. If <see langword="null"/>, a new instance
    /// of <see cref="Random"/> will be created.
    /// </param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this float[,,,] source, float[,,,] secondMatrix, Random? random)
    {
        random ??= new();
        int sourceRows = source.GetLength(0);
        int sourceDim2 = source.GetLength(1);
        int sourceDim3 = source.GetLength(2);
        int sourceDim4 = source.GetLength(3);

        int secondDim2 = secondMatrix.GetLength(1);
        int secondDim3 = secondMatrix.GetLength(2);
        int secondDim4 = secondMatrix.GetLength(3);

        Debug.Assert(sourceRows == secondMatrix.GetLength(0), "Both matrices must have the same number of dim1 to permute them together.");

        for (int sourceRow = sourceRows - 1; sourceRow > 0; sourceRow--)
        {
            int randomRow = random.Next(sourceRow + 1);
            if (sourceRow != randomRow)
            {
                // Swap row i with row j
                for (int d2 = 0; d2 < sourceDim2; d2++)
                {
                    for (int d3 = 0; d3 < sourceDim3; d3++)
                    {
                        for (int d4 = 0; d4 < sourceDim4; d4++)
                        {
                            (source[randomRow, d2, d3, d4], source[sourceRow, d2, d3, d4]) = (source[sourceRow, d2, d3, d4], source[randomRow, d2, d3, d4]);
                        }
                    }
                }
                for (int d2 = 0; d2 < secondDim2; d2++)
                {
                    for (int d3 = 0; d3 < secondDim3; d3++)
                    {
                        for (int d4 = 0; d4 < secondDim4; d4++)
                        {
                            (secondMatrix[randomRow, d2, d3, d4], secondMatrix[sourceRow, d2, d3, d4]) = (secondMatrix[sourceRow, d2, d3, d4], secondMatrix[randomRow, d2, d3, d4]);
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Randomly permutes the dim1 of the specified matrices in place, ensuring that corresponding dim1 in both matrices
    /// remain aligned after permutation.
    /// </summary>
    /// <remarks>
    /// This method performs a Fisher–Yates shuffle on the dim1 of both matrices, maintaining the correspondence between
    /// dim1. This is useful when shuffling paired datasets, such as features and labels, to preserve their alignment.
    /// Both matrices must have the same number of dim1; otherwise, the behavior is undefined.
    /// </remarks>
    /// <param name="source">
    /// The first matrix whose dim1 will be permuted in place. Must have the same number of dim1 as
    /// <paramref name="secondMatrix"/>.
    /// </param>
    /// <param name="secondMatrix">
    /// The second matrix whose dim1 will be permuted in place together with <paramref name="source"/>. Must have the
    /// same number of dim1 as <paramref name="source"/>.
    /// </param>
    /// <param name="random">
    /// The random number generator used to determine the permutation order. If <see langword="null"/>, a new instance
    /// will be created.
    /// </param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWithSetRow(this float[,] source, float[,] secondMatrix, Random? random)
    {
        random ??= new();
        int rows = source.GetLength(0);

        Debug.Assert(rows == secondMatrix.GetLength(0), "Both matrices must have the same number of dim1 to permute them together.");

        for (int i = rows - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            if (i != j)
            {
                float[] tempI = source.GetRow(i);
                source.SetRow(i, source.GetRow(j));
                source.SetRow(j, tempI);

                tempI = secondMatrix.GetRow(i);
                secondMatrix.SetRow(i, secondMatrix.GetRow(j));
                secondMatrix.SetRow(j, tempI);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWithSetRow(this float[,,,] source, float[,] secondMatrix, Random? random)
    {
        random ??= new();
        int dim1 = source.GetLength(0);

        Debug.Assert(dim1 == secondMatrix.GetLength(0), "Both matrices must have the same number of dim1 to permute them together.");

        for (int i = dim1 - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            if (i != j)
            {
                float[,,] tempI3 = source.GetRow(i);
                source.SetRow(i, source.GetRow(j));
                source.SetRow(j, tempI3);

                float[] tempI1 = secondMatrix.GetRow(i);
                secondMatrix.SetRow(i, secondMatrix.GetRow(j));
                secondMatrix.SetRow(j, tempI1);
            }
        }
    }
    */
}
