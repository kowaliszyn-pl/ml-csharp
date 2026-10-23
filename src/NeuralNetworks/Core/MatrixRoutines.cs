// Neural Networks in C♯
// File name: MatrixRoutines.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

public static class MatrixRoutines
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Abs(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = MathF.Abs(source[row, col]);
            }
        }
        return res;
    }

    /// <summary>
    /// Adds a scalar value to each element of the source.
    /// </summary>
    /// <remarks>
    /// This method creates a new array and does not modify the original <paramref name="source"/> array. <para/>
    /// Complexity: O(n * m), where n = dim1 of <paramref name="source"/>, m = dim2 of <paramref name="source"/>.
    /// </remarks>
    /// <param name="source">The two-dimensional array to process.</param>
    /// <param name="scalar">The value to add to each element.</param>
    /// <returns>A new array of the same shape with the result of addition.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Add(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] + scalar;
            }
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Add(this float[,] source, float[,] matrix)
    {
        Debug.Assert(source.GetLength(0) == matrix.GetLength(0) && source.GetLength(1) == matrix.GetLength(1), "Shapes of source and matrix must match.");

        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] + matrix[row, col];
            }
        }
        return res;
    }

    /// <summary>
    /// Adds the specified scalar value to each element of the two-dimensional array in place.
    /// </summary>
    /// <remarks>
    /// This method modifies the contents of the <paramref name="source"/> array directly. The array must be initialized
    /// before calling this method. <para/> Complexity: O(n * m), where n = dim1 of <paramref name="source"/>, m = dim2
    /// of <paramref name="source"/>.
    /// </remarks>
    /// <param name="source">
    /// The two-dimensional array of single-precision floating-point numbers whose elements will be incremented. Cannot
    /// be null.
    /// </param>
    /// <param name="scalar">The scalar value to add to each element of the array.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddInPlace(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                source[row, col] += scalar;
            }
        }
    }

    /// <summary>
    /// Adds the specified scalar value to each element of the three-dimensional array in place.
    /// </summary>
    /// <remarks>
    /// This method modifies the contents of the <paramref name="source"/> array directly. <para/> Complexity: O(a * b *
    /// c) for dimensions of <paramref name="source"/>.
    /// </remarks>
    /// <param name="source">The three-dimensional array whose elements will be incremented.</param>
    /// <param name="scalar">The scalar value to add to each element of the array.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddInPlace(this float[,,] source, float scalar)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    source[i, j, k] += scalar;
                }
            }
        }

    }

    /// <summary>
    /// Adds the specified scalar value to each element of the four-dimensional array in place.
    /// </summary>
    /// <remarks>
    /// This method modifies the contents of the <paramref name="source"/> array directly. <para/> Complexity: O(a * b *
    /// c * d) for dimensions of <paramref name="source"/>.
    /// </remarks>
    /// <param name="source">The four-dimensional array whose elements will be incremented.</param>
    /// <param name="scalar">The scalar value to add to each element of the array.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddInPlace(this float[,,,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        int depth = source.GetLength(2);
        int channels = source.GetLength(3);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                for (int k = 0; k < depth; k++)
                {
                    for (int l = 0; l < channels; l++)
                    {
                        source[i, j, k, l] += scalar;
                    }
                }
            }
        }

    }

    /// <summary>
    /// Adds a row to the current source by elementwise addition with the specified source.
    /// </summary>
    /// <remarks>
    /// This method creates a new array and does not modify the original <paramref name="source"/> array. It adds
    /// <paramref name="vector"/> to each row of <paramref name="source"/> elementwise. <para/> Complexity: O(n * m),
    /// where n = dim1 of <paramref name="source"/>, m = dim2 of <paramref name="source"/>.
    /// </remarks>
    /// <param name="source">The source two-dimensional array.</param>
    /// <param name="vector">
    /// The row values to be added elementwise to each row of the source. Length must equal the number of dim2.
    /// </param>
    /// <returns>A new array containing the elementwise sum of each row with <paramref name="vector"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] AddRow(this float[,] source, float[] vector)
    {
        Debug.Assert(vector.Length == source.GetLength(1));

        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] + vector[col];
            }
        }

        return res;
    }

    /// <summary>
    /// Creates a new source filled with ones, with the same dimensions as the specified source.
    /// </summary>
    /// <param name="source">The source used to determine the dimensions of the new source.</param>
    /// <returns>A new source filled with ones.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] AsOnes(this float[,,,] source)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        res[i, j, k, l] = 1;
                    }
                }
            }
        }

        return res;
    }

    /// <summary>
    /// Creates a new source filled with ones, with the same dimensions as the specified source.
    /// </summary>
    /// <param name="source">The source used to determine the dimensions of the new source.</param>
    /// <returns>A new source filled with ones.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] AsOnes(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = 1;
            }
        }

        return res;
    }

    /// <summary>
    /// Creates a new one-dimensional array filled with ones, with the same length as the specified source.
    /// </summary>
    /// <remarks>
    /// Example: <code> float[] source = new float[] { 2.0f, 3.0f, 4.0f }; var res = source.AsOnes(); // returns new
    /// float[] { 1.0f, 1.0f, 1.0f} </code>
    /// </remarks>
    /// <param name="source">The one-dimensional array used to determine the length of the new array.</param>
    /// <returns>A new one-dimensional array filled with ones.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] AsOnes(this float[] source)
    {
        int length = source.GetLength(0);
        float[] res = new float[length];

        for (int i = 0; i < length; i++)
        {
            res[i] = 1;
        }

        return res;
    }

    /// <summary>
    /// Creates a new two-dimensional array with elements set to 1 with probability <paramref name="onesProbability"/>,
    /// otherwise 0.
    /// </summary>
    /// <param name="source">The array used only for shape (dim1 and dim2).</param>
    /// <param name="onesProbability">Probability of placing 1 in a cell. Must be between 0 and 1.</param>
    /// <param name="random">The random number generator instance.</param>
    /// <returns>A new array with the same shape as <paramref name="source"/> containing zeros and ones.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] AsZeroOnes(this float[,] source, float onesProbability, Random random)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] res = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                if (random.NextDouble() < onesProbability)
                {
                    res[i, j] = 1;
                }
            }
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,] AsZeroOnes(this float[,,] source, float onesProbability, Random random)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        float[,,] res = new float[dim1, dim2, dim3];
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    if (random.NextDouble() < onesProbability)
                    {
                        res[i, j, k] = 1;
                    }
                }
            }
        }
        return res;
    }

    /// <summary>
    /// Creates a new four-dimensional array with elements set to 1 with probability <paramref name="onesProbability"/>,
    /// otherwise 0.
    /// </summary>
    /// <param name="source">The array used only for shape in all four dimensions.</param>
    /// <param name="onesProbability">Probability of placing 1 in a cell. Must be between 0 and 1.</param>
    /// <param name="random">The random number generator instance.</param>
    /// <returns>A new array with the same shape as <paramref name="source"/> containing zeros and ones.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] AsZeroOnes(this float[,,,] source, float onesProbability, Random random)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        if (random.NextDouble() < onesProbability)
                        {
                            res[i, j, k, l] = 1;
                        }
                    }
                }
            }
        }
        return res;
    }

    /// <summary>
    /// Creates a new two-dimensional zero-filled array with the same shape as the source.
    /// </summary>
    /// <param name="source">The array used only for shape.</param>
    /// <returns>A new zero-filled array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] AsZeros(this float[,] source)
        => new float[source.GetLength(0), source.GetLength(1)];

    /// <summary>
    /// Creates a new four-dimensional zero-filled array with the same shape as the source.
    /// </summary>
    /// <param name="source">The array used only for shape.</param>
    /// <returns>A new zero-filled array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] AsZeros(this float[,,,] source)
        => new float[source.GetLength(0), source.GetLength(1), source.GetLength(2), source.GetLength(3)];

    /// <summary>
    /// Returns a new two-dimensional array with each element of the source array limited to the specified minimum and
    /// maximum values.
    /// </summary>
    /// <remarks>
    /// If an element in the source array is less than the specified minimum, the result will contain the minimum value
    /// at that position. If an element is greater than the specified maximum, the result will contain the maximum
    /// value. The original array is not modified.
    /// </remarks>
    /// <param name="source">The two-dimensional array of single-precision floating-point numbers to be clipped.</param>
    /// <param name="min">The minimum value to which elements in the array will be limited.</param>
    /// <param name="max">The maximum value to which elements in the array will be limited.</param>
    /// <returns>
    /// A new two-dimensional array where each element is set to the corresponding value from the source array, limited
    /// to the specified minimum and maximum values.
    /// </returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Clip(this float[,] source, float min, float max)
    {
        Debug.Assert(min <= max, "Minimum value must be less than or equal to maximum value.");

        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = MathF.Max(min, MathF.Min(max, source[row, col]));
            }
        }
        return res;
    }

    /// <summary>
    /// Clips the values of the source in-place between the specified minimum and maximum values.
    /// </summary>
    /// <param name="source">The two-dimensional array to clip.</param>
    /// <param name="min">The minimum value to clip the source elements to.</param>
    /// <param name="max">The maximum value to clip the source elements to.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ClipInPlace(this float[,] source, float min, float max)
    {
        Debug.Assert(min <= max, "Minimum value must be less than or equal to maximum value.");

        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                source[row, col] = MathF.Max(min, MathF.Min(max, source[row, col]));
            }
        }
    }

    /// <summary>
    /// Divides each element of the two-dimensional array by a scalar and returns a new array.
    /// </summary>
    /// <param name="source">The array whose elements will be divided.</param>
    /// <param name="scalar">The divisor.</param>
    /// <returns>A new array containing the division results.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Divide(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] / scalar;
            }
        }

        return res;
    }

    /// <summary>
    /// Divides each element of the two-dimensional array by the corresponding element of another array elementwise.
    /// </summary>
    /// <param name="source">The array whose elements will be divided.</param>
    /// <param name="divisor">The array containing the divisors.</param>
    /// <returns>A new array containing the division results.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] DivideElementwise(this float[,] source, float[,] divisor)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] / divisor[row, col];
            }
        }

        return res;
    }

    /// <summary>
    /// Divides each element of the two-dimensional array by a scalar in place.
    /// </summary>
    /// <param name="source">The array to modify.</param>
    /// <param name="scalar">The divisor.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DivideInPlace(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                source[row, col] /= scalar;
            }
        }
    }

    /// <summary>
    /// Divides each element of a three-dimensional array by a scalar in place.
    /// </summary>
    /// <param name="source">The array to modify.</param>
    /// <param name="scalar">The divisor.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DivideInPlace(this float[,,] source, float scalar)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        for (int d1 = 0; d1 < dim1; d1++)
        {
            for (int d2 = 0; d2 < dim2; d2++)
            {
                for (int d3 = 0; d3 < dim3; d3++)
                {
                    source[d1, d2, d3] /= scalar;
                }
            }
        }
    }

    /// <summary>
    /// Divides each element of the four-dimensional array by a scalar in place.
    /// </summary>
    /// <param name="source">The array to modify.</param>
    /// <param name="scalar">The divisor.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DivideInPlace(this float[,,,] source, float scalar)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int d1 = 0; d1 < dim1; d1++)
        {
            for (int d2 = 0; d2 < dim2; d2++)
            {
                for (int d3 = 0; d3 < dim3; d3++)
                {
                    for (int d4 = 0; d4 < dim4; d4++)
                    {
                        source[d1, d2, d3, d4] /= scalar;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Computes the natural logarithm elementwise and returns a new array.
    /// </summary>
    /// <param name="source">The array whose elements will be transformed.</param>
    /// <returns>A new array with <c>log(x)</c> applied elementwise.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Log(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = MathF.Log(source[i, j]);
            }
        }

        return res;
    }

    /// <summary>
    /// Multiplies each element of the source by a scalar value.
    /// </summary>
    /// <param name="source">The array whose elements will be multiplied.</param>
    /// <param name="scalar">The multiplier.</param>
    /// <returns>A new array with the multiplication results.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Multiply(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] * scalar;
            }
        }

        return res;
    }

    /// <summary>
    /// Multiplies each element of the source by a scalar value.
    /// </summary>
    /// <remarks>
    /// Complexity: O(n * m), where n = dim1 of <paramref name="source"/>, m = dim2 of <paramref name="source"/>
    /// </remarks>
    /// <param name="source">The one-dimensional array to multiply.</param>
    /// <param name="scalar">The multiplier.</param>
    /// <returns>A new array containing the multiplication results.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] Multiply(this float[] source, float scalar)
    {
        int length = source.GetLength(0);
        float[] res = new float[length];
        for (int i = 0; i < length; i++)
        {
            res[i] = source[i] * scalar;
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,] Multiply(this float[,,] source, float scalar)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        float[,,] res = new float[dim1, dim2, dim3];

        for (int d1 = 0; d1 < dim1; d1++)
        {
            for (int d2 = 0; d2 < dim2; d2++)
            {
                for (int d3 = 0; d3 < dim3; d3++)
                {
                    res[d1, d2, d3] = source[d1, d2, d3] * scalar;
                }
            }
        }

        return res;
    }

    /// <summary>
    /// Multiplies each element of the source by a scalar value.
    /// </summary>
    /// <param name="source">The four-dimensional array to multiply.</param>
    /// <param name="scalar">The multiplier.</param>
    /// <returns>A new array with each element multiplied by the scalar value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Multiply(this float[,,,] source, float scalar)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        for (int d1 = 0; d1 < dim1; d1++)
        {
            for (int d2 = 0; d2 < dim2; d2++)
            {
                for (int d3 = 0; d3 < dim3; d3++)
                {
                    for (int d4 = 0; d4 < dim4; d4++)
                    {
                        res[d1, d2, d3, d4] = source[d1, d2, d3, d4] * scalar;
                    }
                }
            }
        }

        return res;
    }

    /// <summary>
    /// Multiplies the current source with another source using the dot product.
    /// </summary>
    /// <remarks>
    /// Complexity: O(n * m * p), where n = dim1 of <paramref name="source"/>, m = shared dimension, p = dim2 of
    /// <paramref name="matrix"/>
    /// </remarks>
    /// <param name="source">Left operand with shape [n, m].</param>
    /// <param name="matrix">Right operand with shape [m, p].</param>
    /// <returns>A new array with shape [n, p] containing the dot product result.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] MultiplyDot(this float[,] source, float[,] matrix)
    {
        Debug.Assert(source.GetLength(1) == matrix.GetLength(0));

        int matrixColumns = matrix.GetLength(1);

        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] res = new float[rows, matrixColumns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < matrixColumns; j++)
            {
                float sum = 0;
                for (int k = 0; k < columns; k++)
                {
                    sum += source[i, k] * matrix[k, j];
                }
                res[i, j] = sum;
            }
        }

        return res;
    }

    /// <summary>
    /// Performs elementwise multiplication between this source and another source.
    /// </summary>
    /// <param name="source">The left operand.</param>
    /// <param name="matrix">The source to multiply elementwise with.</param>
    /// <returns>A new source resulting from the elementwise multiplication.</returns>
    /// <remarks>
    /// Multiplies each element of the source with the corresponding element of another source. If the dimensions of the
    /// two matrices are not the same, the smaller source is broadcasted to match the larger source. If the size of this
    /// source is (a * b), and the size of source is (c * d), then the resulting size is (max(a,c) * max(b,d))
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] MultiplyElementwise(this float[,] source, float[,] matrix)
    {
        int thisRows = source.GetLength(0);
        int thisColumns = source.GetLength(1);
        int matrixRows = matrix.GetLength(0);
        int matrixColumns = matrix.GetLength(1);

        int maxRows = Math.Max(thisRows, matrixRows);
        int maxColumns = Math.Max(thisColumns, matrixColumns);

        Debug.Assert(maxRows % thisRows == 0 && maxRows % matrixRows == 0, "The number of dim1 of one matrix must be a multiple of the other matrix.");
        Debug.Assert(maxColumns % thisColumns == 0 && maxColumns % matrixColumns == 0, "The number of dim2 of one matrix must be a multiple of the other matrix.");

        float[,] res = new float[maxRows, maxColumns];

        for (int row = 0; row < maxRows; row++)
        {
            for (int col = 0; col < maxColumns; col++)
            {
                float thisValue = source[row % thisRows, col % thisColumns];
                float matrixValue = matrix[row % matrixRows, col % matrixColumns];
                res[row, col] = thisValue * matrixValue;
            }
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,] MultiplyElementwise(this float[,,] source, float[,,] matrix)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        int matrixDim1 = matrix.GetLength(0);
        int matrixDim2 = matrix.GetLength(1);
        int matrixDim3 = matrix.GetLength(2);

        int maxDim1 = Math.Max(dim1, matrixDim1);
        int maxDim2 = Math.Max(dim2, matrixDim2);
        int maxDim3 = Math.Max(dim3, matrixDim3);

        float[,,] res = new float[maxDim1, maxDim2, maxDim3];

        for (int d1 = 0; d1 < maxDim1; d1++)
        {
            for (int d2 = 0; d2 < maxDim2; d2++)
            {
                for (int d3 = 0; d3 < maxDim3; d3++)
                {
                    float thisValue = source[d1 % dim1, d2 % dim2, d3 % dim3];
                    float matrixValue = matrix[d1 % matrixDim1, d2 % matrixDim2, d3 % matrixDim3];
                    res[d1, d2, d3] = thisValue * matrixValue;
                }
            }
        }

        return res;

    }

    /// <summary>
    /// Performs elementwise multiplication between this source and another source.
    /// </summary>
    /// <param name="source">The left operand.</param>
    /// <param name="matrix">The source to multiply elementwise with.</param>
    /// <returns>A new source resulting from the elementwise multiplication.</returns>
    /// <remarks>
    /// Multiplies each element of the source with the corresponding element of another source. If the dimensions of the
    /// two matrices are not the same, the smaller source is broadcasted to match the larger source. If the size of this
    /// source is (a * b * c * d), and the size of source is (e * f * g * h), then the resulting size is (max(a,e) *
    /// max(b,f) * max(c,g) * max(d,h))
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] MultiplyElementwise(this float[,,,] source, float[,,,] matrix)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        int matrixDim1 = matrix.GetLength(0);
        int matrixDim2 = matrix.GetLength(1);
        int matrixDim3 = matrix.GetLength(2);
        int matrixDim4 = matrix.GetLength(3);

        int maxDim1 = Math.Max(dim1, matrixDim1);
        int maxDim2 = Math.Max(dim2, matrixDim2);
        int maxDim3 = Math.Max(dim3, matrixDim3);
        int maxDim4 = Math.Max(dim4, matrixDim4);

        float[,,,] res = new float[maxDim1, maxDim2, maxDim3, maxDim4];

        for (int d1 = 0; d1 < maxDim1; d1++)
        {
            for (int d2 = 0; d2 < maxDim2; d2++)
            {
                for (int d3 = 0; d3 < maxDim3; d3++)
                {
                    for (int d4 = 0; d4 < maxDim4; d4++)
                    {
                        float thisValue = source[d1 % dim1, d2 % dim2, d3 % dim3, d4 % dim4];
                        float matrixValue = matrix[d1 % matrixDim1, d2 % matrixDim2, d3 % matrixDim3, d4 % matrixDim4];
                        res[d1, d2, d3, d4] = thisValue * matrixValue;
                    }
                }
            }
        }

        return res;

    }

    /// <summary>
    /// Performs elementwise multiplication between a vector and a matrix, with broadcasting across dim1.
    /// </summary>
    /// <param name="source">The one-dimensional array treated as a row vector.</param>
    /// <param name="matrix">The two-dimensional array to multiply elementwise.</param>
    /// <returns>
    /// A new two-dimensional array of shape [dim1 of <paramref name="matrix"/>, max(dim2)] containing the elementwise
    /// product.
    /// </returns>
    /// <remarks>
    /// If the dimensions are not the same, the smaller array is broadcasted across dim2.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] MultiplyElementwise(this float[] source, float[,] matrix)
    {
        int thisColumns = source.GetLength(0);
        int matrixRows = matrix.GetLength(0);
        int matrixColumns = matrix.GetLength(1);

        int maxColumns = Math.Max(thisColumns, matrixColumns);

        Debug.Assert(maxColumns % thisColumns == 0 && maxColumns % matrixColumns == 0, "The number of dim2 of one matrix must be a multiple of the other matrix.");

        float[,] res = new float[matrixRows, maxColumns];

        for (int row = 0; row < matrixRows; row++)
        {
            for (int col = 0; col < maxColumns; col++)
            {
                float thisValue = source[col % thisColumns];
                float matrixValue = matrix[row % matrixRows, col % matrixColumns]; // TODO: The row % matrixRows modulo is unnecessary because row already iterates from 0 to matrixRows - 1. 
                res[row, col] = thisValue * matrixValue;
            }
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] MultiplyElementwise(this float[,] source, float[] matrix)
    {
        int thisRows = source.GetLength(0);
        int thisColumns = source.GetLength(1);
        int matrixLength = matrix.GetLength(0);
        int maxColumns = Math.Max(thisColumns, matrixLength);

        Debug.Assert(maxColumns % thisColumns == 0 && maxColumns % matrixLength == 0, "The number of dim2 of one matrix must be a multiple of the other matrix.");

        float[,] res = new float[thisRows, maxColumns];
        for (int row = 0; row < thisRows; row++)
        {
            for (int col = 0; col < maxColumns; col++)
            {
                float thisValue = source[row, col % thisColumns];
                float matrixValue = matrix[col % matrixLength];
                res[row, col] = thisValue * matrixValue;
            }
        }
        return res;
    }

    /// <summary>
    /// Raises each element of the source to the specified power.
    /// </summary>
    /// <param name="source">The two-dimensional array whose elements will be exponentiated.</param>
    /// <param name="scalar">The exponent (integer).</param>
    /// <returns>A new array with each element raised to <paramref name="scalar"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Power(this float[,] source, int scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = MathF.Pow(source[row, col], scalar);
            }
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Power(this float[,,,] source, int scalar)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        for (int d1 = 0; d1 < dim1; d1++)
        {
            for (int d2 = 0; d2 < dim2; d2++)
            {
                for (int d3 = 0; d3 < dim3; d3++)
                {
                    for (int d4 = 0; d4 < dim4; d4++)
                    {
                        res[d1, d2, d3, d4] = MathF.Pow(source[d1, d2, d3, d4], scalar);
                    }
                }
            }
        }

        return res;
    }

    /// <summary>
    /// Subtracts the elements of the specified four-dimensional array from the current four-dimensional array.
    /// </summary>
    /// <param name="source">The minuend array.</param>
    /// <param name="matrix">The subtrahend array. Must have the same shape as <paramref name="source"/>.</param>
    /// <returns>A new array containing the elementwise difference.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Subtract(this float[,,,] source, float[,,,] matrix)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        for (int d1 = 0; d1 < dim1; d1++)
        {
            for (int d2 = 0; d2 < dim2; d2++)
            {
                for (int d3 = 0; d3 < dim3; d3++)
                {
                    for (int d4 = 0; d4 < dim4; d4++)
                    {
                        res[d1, d2, d3, d4] = source[d1, d2, d3, d4] - matrix[d1, d2, d3, d4];
                    }
                }
            }
        }

        return res;
    }

    /// <summary>
    /// Subtracts the elements of the specified source from the current source.
    /// </summary>
    /// <param name="source">The two-dimensional minuend array.</param>
    /// <param name="matrix">The two-dimensional subtrahend array. Must have the same shape.</param>
    /// <returns>A new array containing the elementwise difference.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Subtract(this float[,] source, float[,] matrix)
    {
        Debug.Assert(source.GetLength(0) == matrix.GetLength(0));
        Debug.Assert(source.GetLength(1) == matrix.GetLength(1));

        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = source[i, j] - matrix[i, j];
            }
        }

        return res;
    }

    /// <summary>
    /// Subtracts the elements of the specified one-dimensional array from the current one-dimensional array.
    /// </summary>
    /// <param name="source">The minuend array.</param>
    /// <param name="matrix">The subtrahend array. Must have the same length.</param>
    /// <returns>A new array containing the elementwise difference.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] Subtract(this float[] source, float[] matrix)
    {
        Debug.Assert(source.GetLength(0) == matrix.GetLength(0));
        int length = source.GetLength(0);
        float[] res = new float[length];
        for (int i = 0; i < length; i++)
        {
            res[i] = source[i] - matrix[i];
        }
        return res;
    }

    /// <summary>
    /// Transposes the matrix by swapping its rows and columns.
    /// </summary>
    /// <param name="source">The two-dimensional array to transpose.</param>
    /// <returns>A new array with shape [columns, rows].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Transpose(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] array = new float[columns, rows];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[j, i] = source[i, j];
            }
        }

        return array;
    }
}
