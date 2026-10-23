// Neural Networks in C♯
// File name: ArrayExtensions.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

/// <summary>
/// Provides extension methods for performing mathematical and utility operations on multidimensional arrays of type
/// <see cref="float"/>. These methods enable elementwise arithmetic, statistical calculations, source manipulation, and
/// other common operations for arrays and matrices.
/// </summary>
/// <remarks>
/// The <see cref="ArrayExtensions"/> class includes methods for both in-place and out-of-place operations, supporting
/// 1D, 2D, and 4D float arrays. It offers functionality such as elementwise addition, multiplication, subtraction,
/// standardization, transposition, and application of activation functions (e.g., sigmoid, softmax, tanh). Methods are
/// designed to simplify array manipulation in numerical and machine learning scenarios. Thread safety is not
/// guaranteed; callers should synchronize access if arrays are shared across threads.
/// </remarks>
public static class ArrayExtensions
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
    /// <paramref name="matrix"/> to each row of <paramref name="source"/> elementwise. <para/> Complexity: O(n * m),
    /// where n = dim1 of <paramref name="source"/>, m = dim2 of <paramref name="source"/>.
    /// </remarks>
    /// <param name="source">The source two-dimensional array.</param>
    /// <param name="matrix">
    /// The row values to be added elementwise to each row of the source. Length must equal the number of dim2.
    /// </param>
    /// <returns>A new array containing the elementwise sum of each row with <paramref name="matrix"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] AddRow(this float[,] source, float[] matrix)
    {
        Debug.Assert(matrix.Length == source.GetLength(1));

        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] + matrix[col];
            }
        }

        return res;
    }

    /// <summary>
    /// Computes the index of the maximum value for each row.
    /// </summary>
    /// <param name="source">The array to evaluate.</param>
    /// <returns>The index of the maximum value in the array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Argmax(this float[] source)
    {
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
    /// Checks whether two one-dimensional arrays have the same shape (length).
    /// </summary>
    /// <param name="source">The first array.</param>
    /// <param name="matrix">The second array.</param>
    /// <returns>True if both arrays have equal length; otherwise false.</returns>
    public static bool HasSameShape(this float[] source, float[] matrix)
        => source.GetLength(0) == matrix.GetLength(0);

    /// <summary>
    /// Checks whether two two-dimensional arrays have the same shape (dim1 and dim2).
    /// </summary>
    /// <param name="source">The first array.</param>
    /// <param name="matrix">The second array.</param>
    /// <returns>True if both arrays have equal dim1 and dim2; otherwise false.</returns>
    public static bool HasSameShape(this float[,] source, float[,] matrix)
        => source.GetLength(0) == matrix.GetLength(0)
            && source.GetLength(1) == matrix.GetLength(1);

    public static bool HasSameShape(this float[,,] source, float[,,] matrix)
        => source.GetLength(0) == matrix.GetLength(0)
            && source.GetLength(1) == matrix.GetLength(1)
            && source.GetLength(2) == matrix.GetLength(2);

    /// <summary>
    /// Checks whether two four-dimensional arrays have the same shape across all dimensions.
    /// </summary>
    /// <param name="source">The first array.</param>
    /// <param name="matrix">The second array.</param>
    /// <returns>True if all corresponding dimensions are equal; otherwise false.</returns>
    public static bool HasSameShape(this float[,,,] source, float[,,,] matrix)
        => source.GetLength(0) == matrix.GetLength(0)
            && source.GetLength(1) == matrix.GetLength(1)
            && source.GetLength(2) == matrix.GetLength(2)
            && source.GetLength(3) == matrix.GetLength(3);

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

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
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

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
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

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
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

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
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
    public static float Mean(this float[,,,] source) => source.Sum() / source.Length;

    /// <summary>
    /// Calculates the mean of each column in the source.
    /// </summary>
    /// <param name="source">The two-dimensional array to process.</param>
    /// <returns>A one-dimensional array where each element is the mean of a column.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] MeanByColumn(this float[,] source)
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
            res[col] = sum / rows;
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] MultiplyByTanhDerivative(this float[,,,] outputGradient, float[,,,] output)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Tanh function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Tanh function tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) is 1 - tanh(x)^2.
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * (1 - tanh(x)^2).
        // The elementwise multiplication of the output gradient and the derivative of the Tanh function is returned as the input gradient.
        // tanh(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient

        int d0 = outputGradient.GetLength(0);
        int d1 = outputGradient.GetLength(1);
        int d2 = outputGradient.GetLength(2);
        int d3 = outputGradient.GetLength(3);

        Debug.Assert(d0 > 0 && d1 > 0 && d2 > 0 && d3 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(output.GetLength(0) == d0 && output.GetLength(1) == d1 && output.GetLength(2) == d2 && output.GetLength(3) == d3, "Shapes of outputGradient and output must match for elementwise operations.");

        float[,,,] result = new float[d0, d1, d2, d3];

        for (int i = 0; i < d0; i++)
        {
            for (int j = 0; j < d1; j++)
            {
                for (int k = 0; k < d2; k++)
                {
                    for (int l = 0; l < d3; l++)
                    {
                        float y = output[i, j, k, l];
                        float dy = outputGradient[i, j, k, l];
                        float tanhDerivative = 1f - (y * y);
                        result[i, j, k, l] = dy * tanhDerivative;
                    }
                }
            }
        }
        return result;
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

        int maxDim1 = Math.Max(dim1, matrix.GetLength(0));
        int maxDim2 = Math.Max(dim2, matrix.GetLength(1));
        int maxDim3 = Math.Max(dim3, matrix.GetLength(2));

        float[,,] res = new float[maxDim1, maxDim2, maxDim3];

        for (int d1 = 0; d1 < maxDim1; d1++)
        {
            for (int d2 = 0; d2 < maxDim2; d2++)
            {
                for (int d3 = 0; d3 < maxDim3; d3++)
                {
                    float thisValue = source[d1 % dim1, d2 % dim2, d3 % dim3];
                    float matrixValue = matrix[d1 % matrix.GetLength(0), d2 % matrix.GetLength(1), d3 % matrix.GetLength(2)]; // TODO: Refactor to avoid repeated calls to GetLength in the inner loop.
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

        int maxDim1 = Math.Max(dim1, matrix.GetLength(0));
        int maxDim2 = Math.Max(dim2, matrix.GetLength(1));
        int maxDim3 = Math.Max(dim3, matrix.GetLength(2));
        int maxDim4 = Math.Max(dim4, matrix.GetLength(3));

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
                        float matrixValue = matrix[d1 % matrix.GetLength(0), d2 % matrix.GetLength(1), d3 % matrix.GetLength(2), d4 % matrix.GetLength(3)];
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
                float matrixValue = matrix[row % matrixRows, col % matrixColumns];
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

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
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

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
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

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
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

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
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