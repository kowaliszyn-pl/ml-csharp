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
