// Neural Networks in C♯
// File name: ArrayExtensions.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NeuralNetworks.Core;

/// <summary>
/// Provides extension methods for performing mathematical and utility operations on multidimensional arrays of type
/// <see cref="float"/>. These methods enable elementwise arithmetic, statistical calculations, source manipulation, and
/// other common operations for arrays and matrices.
/// </summary>
/// <remarks>The <see cref="ArrayExtensions"/> class includes methods for both in-place and out-of-place
/// operations, supporting 1D, 2D, and 4D float arrays. It offers functionality such as elementwise addition,
/// multiplication, subtraction, standardization, transposition, and application of activation functions (e.g., sigmoid,
/// softmax, tanh). Methods are designed to simplify array manipulation in numerical and machine learning scenarios.
/// Thread safety is not guaranteed; callers should synchronize access if arrays are shared across threads.</remarks>
public static class ArrayExtensions
{
    /// <summary>
    /// Adds a scalar value to each element of the source.
    /// </summary>
    /// <remarks>
    /// This method creates a new array and does not modify the original <paramref name="source"/> array.
    /// <para/>
    /// Complexity: O(n * m), where n = rows of <paramref name="source"/>, m = columns of <paramref name="source"/>.
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

    /// <summary>
    /// Adds the specified scalar value to each element of the two-dimensional array in place.
    /// </summary>
    /// <remarks>
    /// This method modifies the contents of the <paramref name="source"/> array directly. The array
    /// must be initialized before calling this method.
    /// <para/>
    /// Complexity: O(n * m), where n = rows of <paramref name="source"/>, m = columns of <paramref name="source"/>.
    /// </remarks>
    /// <param name="source">The two-dimensional array of single-precision floating-point numbers whose elements will be incremented. Cannot be null.
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
    /// Adds the specified scalar value to each element of the four-dimensional array in place.
    /// </summary>
    /// <remarks>
    /// This method modifies the contents of the <paramref name="source"/> array directly.
    /// <para/>
    /// Complexity: O(a * b * c * d) for dimensions of <paramref name="source"/>.
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
    /// This method creates a new array and does not modify the original <paramref name="source"/> array. It adds <paramref name="matrix"/> to each row of <paramref name="source"/> elementwise.
    /// <para/>
    /// Complexity: O(n * m), where n = rows of <paramref name="source"/>, m = columns of <paramref name="source"/>.
    /// </remarks>
    /// <param name="source">The source two-dimensional array.</param>
    /// <param name="matrix">The row values to be added elementwise to each row of the source. Length must equal the number of columns.</param>
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
    /// <param name="source">The two-dimensional array to evaluate.</param>
    /// <returns>An array of length equal to the number of rows, where each element is the column index of the maximum value in that row.</returns>
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
    /// Example: 
    /// <code>
    /// float[] source = new float[] { 2.0f, 3.0f, 4.0f };
    /// var res = source.AsOnes(); // returns new float[] { 1.0f, 1.0f, 1.0f }
    /// </code>
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
    /// Creates a new two-dimensional array with elements set to 1 with probability <paramref name="onesProbability"/>, otherwise 0.
    /// </summary>
    /// <param name="source">The array used only for shape (rows and columns).</param>
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

    /// <summary>
    /// Creates a new four-dimensional array with elements set to 1 with probability <paramref name="onesProbability"/>, otherwise 0.
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
    /// <remarks>If an element in the source array is less than the specified minimum, the result will contain
    /// the minimum value at that position. If an element is greater than the specified maximum, the result will contain
    /// the maximum value. The original array is not modified.</remarks>
    /// <param name="source">The two-dimensional array of single-precision floating-point numbers to be clipped.</param>
    /// <param name="min">The minimum value to which elements in the array will be limited.</param>
    /// <param name="max">The maximum value to which elements in the array will be limited.</param>
    /// <returns>A new two-dimensional array where each element is set to the corresponding value from the source array, limited
    /// to the specified minimum and maximum values.</returns>
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
    /// Gets a submatrix containing the specified column from the current source. The shape is [rows, 1].
    /// </summary>
    /// <param name="source">The array to slice.</param>
    /// <param name="column">The zero-based column index.</param>
    /// <returns>A new [rows, 1] array representing the selected column.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] GetColumn(this float[,] source, int column)
    {
        int rows = source.GetLength(0);

        // Create an array to store the column.
        float[,] res = new float[rows, 1];

        for (int i = 0; i < rows; i++)
        {
            // Access each element in the specified column.
            res[i, 0] = source[i, column];
        }

        return res;
    }

    /// <summary>
    /// Gets a submatrix containing the specified range of dim2 from the current source. The shape is [rows, range].
    /// </summary>
    /// <param name="source">The array to slice.</param>
    /// <param name="range">The range specifying the subset of columns.</param>
    /// <returns>A new array with [rows, selectedColumns].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] GetColumns(this float[,] source, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(source.GetLength(1));

        int rows = source.GetLength(0);
        float[,] res = new float[rows, length];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < length; j++)
            {
                res[i, j] = source[i, j + offset];
            }
        }

        return res;
    }

    /// <summary>
    /// Gets a row from the source.
    /// </summary>
    /// <param name="source">The two-dimensional array to slice.</param>
    /// <param name="row">The index of the row to retrieve.</param>
    /// <returns>A new <see cref="float[]"/> representing the specified row.</returns>
    /// <remarks>
    /// The returned row is a new instance and has the same number of columns as the original source.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] GetRow(this float[,] source, int row)
    {
        int columns = source.GetLength(1);

        // Create an array to store the row.
        float[] res = new float[columns];
        for (int i = 0; i < columns; i++)
        {
            // Access each element in the specified row.
            res[i] = source[row, i];
        }

        return res;
    }

    /// <summary>
    /// Gets a three-dimensional slice (row) from a four-dimensional array at the specified first-dimension index.
    /// </summary>
    /// <param name="source">The four-dimensional array to slice.</param>
    /// <param name="row">The zero-based index along dimension 0.</param>
    /// <returns>A new three-dimensional array with shape [dim2, dim3, dim4] containing the selected row.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,] GetRow(this float[,,,] source, int row)
    {
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        // Create an array to store the row.
        float[,,] res = new float[dim2, dim3, dim4];
        for (int i = 0; i < dim2; i++)
        {
            for (int j = 0; j < dim3; j++)
            {
                for (int k = 0; k < dim4; k++)
                {
                    // Access each element in the specified row.
                    res[i, j, k] = source[row, i, j, k];
                }
            }
        }

        return res;

    }

    /// <summary>
    /// Gets the specified row as a two-dimensional array with shape [1, columns].
    /// </summary>
    /// <param name="source">The two-dimensional array to slice.</param>
    /// <param name="row">The zero-based row index.</param>
    /// <returns>A new [1, columns] array containing the row values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] GetRowAs2D(this float[,] source, int row)
    {
        int columns = source.GetLength(1);
        // Create an array to store the row.
        float[,] res = new float[1, columns];
        for (int i = 0; i < columns; i++)
        {
            // Access each element in the specified row.
            res[0, i] = source[row, i];
        }
        return res;
    }

    /// <summary>
    /// Gets a submatrix containing the specified range of rows from the current source.
    /// </summary>
    /// <param name="source">The array to slice.</param>
    /// <param name="range">The range of rows to retrieve.</param>
    /// <returns>A new <see cref="float[,]"/> object representing the submatrix.</returns>
    /// <remarks>
    /// The returned rows are a new instance of the <see cref="float[,]"/> class and have the same number of columns as the original source.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] GetRows(this float[,] source, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(source.GetLength(0));

        int columns = source.GetLength(1);
        float[,] res = new float[length, columns];

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = source[i + offset, j];
            }
        }

        return res;
    }

    /// <summary>
    /// Gets a contiguous range of rows (dimension 0) from a four-dimensional array.
    /// </summary>
    /// <param name="source">The four-dimensional array to slice.</param>
    /// <param name="range">The range applied to dimension 0.</param>
    /// <returns>A new four-dimensional array containing the selected rows.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] GetRows(this float[,,,] source, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(source.GetLength(0));

        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[length, dim2, dim3, dim4];

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        res[i, j, k, l] = source[i + offset, j, k, l];
                    }
                }
            }
        }

        return res;

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
    /// Checks whether two two-dimensional arrays have the same shape (rows and columns).
    /// </summary>
    /// <param name="source">The first array.</param>
    /// <param name="matrix">The second array.</param>
    /// <returns>True if both arrays have equal rows and columns; otherwise false.</returns>
    public static bool HasSameShape(this float[,] source, float[,] matrix)
        => source.GetLength(0) == matrix.GetLength(0)
            && source.GetLength(1) == matrix.GetLength(1);

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] LeakyReLU(this float[,] source, float alpha = 0.01f, float beta = 1f)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                float value = source[i, j];
                res[i, j] = value >= 0 ? value * beta : value * alpha;
            }
        }
        return res;
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
    /// Complexity: O(n * m), where n = rows of <paramref name="source"/>, m = columns of <paramref name="source"/>
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
        Debug.Assert(output.GetLength(0) != d0 && output.GetLength(1) != d1 && output.GetLength(2) != d2 && output.GetLength(3) != d3, "Shapes of outputGradient and output must match for elementwise operations.");

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] MultiplyByTanhDerivativeSpan(this float[,,,] outputGradient, float[,,,] output)
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
        Debug.Assert(output.GetLength(0) != d0 && output.GetLength(1) != d1 && output.GetLength(2) != d2 && output.GetLength(3) != d3, "Shapes of outputGradient and output must match for elementwise operations.");

        float[,,,] result = new float[d0, d1, d2, d3];

        ref float ogRef = ref outputGradient[0, 0, 0, 0];
        ref float outRef = ref output[0, 0, 0, 0];
        ref float resRef = ref result[0, 0, 0, 0];

        ReadOnlySpan<float> ogSpan = MemoryMarshal.CreateReadOnlySpan(ref ogRef, outputGradient.Length);
        ReadOnlySpan<float> outSpan = MemoryMarshal.CreateReadOnlySpan(ref outRef, output.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref resRef, result.Length);

        for (int i = 0; i < resSpan.Length; i++)
        {
            float y = outSpan[i];
            float dy = ogSpan[i];
            resSpan[i] = dy * (1f - (y * y));
        }

        return result;
    }

    /// <summary>
    /// Multiplies the current source with another source using the dot product.
    /// </summary>
    /// <remarks>
    /// Complexity: O(n * m * p), where n = rows of <paramref name="source"/>, m = shared dimension, p = columns of <paramref name="matrix"/>
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
    /// Multiplies each element of the source with the corresponding element of another source.
    /// If the dimensions of the two matrices are not the same, the smaller source is broadcasted to match the larger source.
    /// If the size of this source is (a * b), and the size of source is (c * d), then the resulting size is (max(a,c) * max(b,d))
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

        Debug.Assert(maxRows % thisRows == 0 && maxRows % matrixRows == 0, "The number of rows of one matrix must be a multiple of the other matrix.");
        Debug.Assert(maxColumns % thisColumns == 0 && maxColumns % matrixColumns == 0, "The number of columns of one matrix must be a multiple of the other matrix.");

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

    /// <summary>
    /// Performs elementwise multiplication between this source and another source.
    /// </summary>
    /// <param name="source">The left operand.</param>
    /// <param name="matrix">The source to multiply elementwise with.</param>
    /// <returns>A new source resulting from the elementwise multiplication.</returns>
    /// <remarks>
    /// Multiplies each element of the source with the corresponding element of another source.
    /// If the dimensions of the two matrices are not the same, the smaller source is broadcasted to match the larger source.
    /// If the size of this source is (a * b * c * d), and the size of source is (e * f * g * h), then the resulting size is (max(a,e) * max(b,f) * max(c,g) * max(d,h))
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
    /// Performs elementwise multiplication between a vector and a matrix, with broadcasting across rows.
    /// </summary>
    /// <param name="source">The one-dimensional array treated as a row vector.</param>
    /// <param name="matrix">The two-dimensional array to multiply elementwise.</param>
    /// <returns>A new two-dimensional array of shape [rows of <paramref name="matrix"/>, max(columns)] containing the elementwise product.</returns>
    /// <remarks>
    /// If the dimensions are not the same, the smaller array is broadcasted across columns.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] MultiplyElementwise(this float[] source, float[,] matrix)
    {
        int thisColumns = source.GetLength(0);
        int matrixRows = matrix.GetLength(0);
        int matrixColumns = matrix.GetLength(1);

        int maxColumns = Math.Max(thisColumns, matrixColumns);

        Debug.Assert(maxColumns % thisColumns == 0 && maxColumns % matrixColumns == 0, "The number of columns of one matrix must be a multiple of the other matrix.");

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

    /// <summary>
    /// Randomly permutes the rows of the source in-place using the specified seed. It uses the Fisher-Yates shuffle algorithm.
    /// </summary>
    /// <remarks>
    /// Complexity: O(n * m), where n = rows, m = columns.
    /// </remarks>
    /// <param name="source">The two-dimensional array whose rows will be permuted.</param>
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
    /// Randomly permutes the rows of the source in-place using the provided random instance (Fisher-Yates shuffle).
    /// </summary>
    /// <param name="source">The two-dimensional array whose rows will be permuted.</param>
    /// <param name="random">The random number generator. If null, a new instance is created.</param>
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
    /// Randomly permutes the rows of the specified matrices in place, ensuring that corresponding rows in both matrices
    /// remain aligned.
    /// </summary>
    /// <remarks>This method performs an in-place permutation of the rows of both matrices, maintaining the
    /// correspondence between rows. This is useful when shuffling paired data, such as features and labels, for machine
    /// learning tasks. The operation modifies the input matrices directly.
    /// <para/>
    /// This method is the quickest for permuting two 2D matrices together.
    /// </remarks>
    /// <param name="source">The first matrix whose rows will be permuted. Must have the same number of rows as <paramref
    /// name="secondMatrix"/>.</param>
    /// <param name="secondMatrix">The second matrix whose rows will be permuted in tandem with <paramref name="source"/>. Must have the same
    /// number of rows as <paramref name="source"/>.</param>
    /// <param name="random">The random number generator used to determine the permutation order. If <see langword="null"/>, a new instance
    /// will be created.</param>
    public static void PermuteInPlaceTogetherWith(this float[,] source, float[,] secondMatrix, Random? random)
    {
        random ??= new();
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        int secondColumns = secondMatrix.GetLength(1);

        Debug.Assert(rows == secondMatrix.GetLength(0), "Both matrices must have the same number of rows to permute them together.");

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

    /// <summary>
    /// Randomly permutes the rows of the specified four-dimensional array and the corresponding rows of the second
    /// matrix in place, ensuring that both arrays are shuffled together using the same permutation.
    /// </summary>
    /// <remarks>Both <paramref name="source"/> and <paramref name="secondMatrix"/> must have the same number
    /// of rows; otherwise, the method will not perform a valid permutation. The permutation is performed in place and
    /// affects the original arrays. This method is useful for maintaining alignment between related datasets when
    /// shuffling.</remarks>
    /// <param name="source">The four-dimensional array whose rows will be permuted in place. The first dimension represents the rows to be
    /// shuffled.</param>
    /// <param name="secondMatrix">The two-dimensional matrix whose rows will be permuted in place together with the rows of <paramref
    /// name="source"/>. Must have the same number of rows as <paramref name="source"/>.</param>
    /// <param name="random">The random number generator used to determine the permutation order. If <see langword="null"/>, a new instance
    /// of <see cref="Random"/> will be created.</param>
    public static void PermuteInPlaceTogetherWith(this float[,,,] source, float[,] secondMatrix, Random? random)
    {
        random ??= new();
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);

        int secondColumns = secondMatrix.GetLength(1);

        Debug.Assert(dim1 == secondMatrix.GetLength(0), "Both matrices must have the same number of rows to permute them together.");

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
    /// Randomly permutes the rows of the specified matrices in place, ensuring that corresponding rows in both matrices
    /// remain aligned after permutation.
    /// </summary>
    /// <remarks>This method performs a Fisher–Yates shuffle on the rows of both matrices, maintaining the
    /// correspondence between rows. This is useful when shuffling paired datasets, such as features and labels, to
    /// preserve their alignment. Both matrices must have the same number of rows; otherwise, the behavior is
    /// undefined.</remarks>
    /// <param name="source">The first matrix whose rows will be permuted in place. Must have the same number of rows as <paramref
    /// name="secondMatrix"/>.</param>
    /// <param name="secondMatrix">The second matrix whose rows will be permuted in place together with <paramref name="source"/>. Must have the
    /// same number of rows as <paramref name="source"/>.</param>
    /// <param name="random">The random number generator used to determine the permutation order. If <see langword="null"/>, a new instance
    /// will be created.</param>
    public static void PermuteInPlaceTogetherWithSetRow(this float[,] source, float[,] secondMatrix, Random? random)
    {
        random ??= new();
        int rows = source.GetLength(0);

        Debug.Assert(rows == secondMatrix.GetLength(0), "Both matrices must have the same number of rows to permute them together.");

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

    public static void PermuteInPlaceTogetherWithSetRow(this float[,,,] source, float[,] secondMatrix, Random? random)
    {
        random ??= new();
        int dim1 = source.GetLength(0);

        Debug.Assert(dim1 == secondMatrix.GetLength(0), "Both matrices must have the same number of rows to permute them together.");

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
    /// Applies the rectified linear unit (ReLU) activation function to each element of the specified 2D array.
    /// </summary>
    /// <remarks>The ReLU function sets all negative values to zero and multiplies non-negative values by the
    /// specified beta. The original array is not modified.</remarks>
    /// <param name="source">The two-dimensional array of single-precision floating-point values to which the ReLU function is applied.</param>
    /// <param name="beta">An optional scaling factor applied to non-negative values. The default is 1.0.</param>
    /// <returns>A new two-dimensional array where each element is the result of applying the ReLU function to the corresponding
    /// element in the source array.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] ReLU(this float[,] source, float beta = 1f)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                float value = source[i, j];
                res[i, j] = value >= 0 ? value * beta : 0;
            }
        }
        return res;
    }

    /// <summary>
    /// Sets the values of a specific row in the source.
    /// </summary>
    /// <param name="source">The two-dimensional array to modify.</param>
    /// <param name="rowIndex">The index of the row to set.</param>
    /// <param name="row">The array containing the values to set. Length must equal the number of columns.</param>
    /// <exception cref="System.Diagnostics.Debug">Asserts when the row index is out of bounds or lengths mismatch.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SetRow(this float[,] source, int rowIndex, float[] row)
    {
        Debug.Assert(rowIndex >= 0 && rowIndex < source.GetLength(0), "Row index out of bounds.");
        Debug.Assert(row.GetLength(0) == source.GetLength(1), "Number of columns must be equal to number of columns.");

        int columns = source.GetLength(1);
        for (int col = 0; col < columns; col++)
        {
            source[rowIndex, col] = row[col];
        }
    }

    /// <summary>
    /// Sets the three-dimensional row slice at the specified index in a four-dimensional array.
    /// </summary>
    /// <param name="source">The four-dimensional array to modify.</param>
    /// <param name="rowIndex">The zero-based index along dimension 0 to set.</param>
    /// <param name="row">A three-dimensional array with shape [dim2, dim3, dim4] providing values.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SetRow(this float[,,,] source, int rowIndex, float[,,] row)
    {
        Debug.Assert(rowIndex >= 0 && rowIndex < source.GetLength(0), "Row index out of bounds.");

        //int dim1 = source.GetLength(0);
        int cols = source.GetLength(1);
        int depths = source.GetLength(2);
        int channels = source.GetLength(3);

        for (int col = 0; col < cols; col++)
        {
            for (int depth = 0; depth < depths; depth++)
            {
                for (int channel = 0; channel < channels; channel++)
                {
                    source[rowIndex, col, depth, channel] = row[col, depth, channel];
                }
            }
        }
    }

    /// <summary>
    /// Applies the sigmoid function to each element of the source.
    /// </summary>
    /// <returns>A new source with each element transformed by the sigmoid function with the same dimensions as the original source.</returns>
    /// <param name="source">The two-dimensional array to transform.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Sigmoid(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = 1 / (1 + MathF.Exp(-source[i, j]));
            }
        }

        return res;
    }

    /// <summary>
    /// Calculates the derivative of the sigmoid function for each element of the source.
    /// </summary>
    /// <remarks>
    /// The derivative of the sigmoid function is calculated as: sigmoid(x) * (1 - sigmoid(x)).
    /// </remarks>
    /// <returns>A new source with each element transformed by the derivative of the sigmoid function with the same dimensions as the original source.</returns>
    /// <param name="source">The two-dimensional array to transform.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] SigmoidDerivative(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                float sigmoid = 1 / (1 + MathF.Exp(-source[i, j]));
                res[i, j] = sigmoid * (1 - sigmoid);
            }
        }

        return res;
    }

    /// <summary>
    /// Applies the softmax function to the source.
    /// </summary>
    /// <returns>A new source with softmax-applied values.</returns>
    /// <remarks>Softmax formula: <c>exp(x) / sum(exp(x))</c>.</remarks>
    /// <param name="source">The two-dimensional array to transform (softmax applied per row).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Softmax(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        float[,] expCache = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                expCache[i, j] = MathF.Exp(source[i, j]);
            }

            float sum = 0;
            for (int j = 0; j < columns; j++)
            {
                sum += expCache[i, j];
            }

            for (int j = 0; j < columns; j++)
            {
                res[i, j] = expCache[i, j] / sum;
            }
        }

        Debug.Assert(res.Cast<float>().All(x => !float.IsNaN(x)), "There should be no NaN values");

        return res;
    }

    /// <summary>
    /// Applies the softmax function (with log-sum-exp trick) to the source.
    /// </summary>
    /// <remarks>
    /// The trick improves numerical stability by subtracting the maximum value in each row before exponentiation. This prevents overflow issues when dealing with large input values.
    /// </remarks>
    /// <returns>A new source with softmax-applied values.</returns>
    /// <param name="source">The two-dimensional array to transform (log-sum-exp softmax applied per row).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] SoftmaxLogSumExp(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++) // dim1 = batch size (obervations)
        {
            float max = source[i, 0];
            for (int j = 1; j < columns; j++)
            {
                max = MathF.Max(max, source[i, j]);
            }

            float sum = 0;
            for (int j = 0; j < columns; j++)
            {
                sum += MathF.Exp(source[i, j] - max);
            }

            float logSumExp = max + MathF.Log(sum);

            for (int j = 0; j < columns; j++)
            {
                res[i, j] = source[i, j] - logSumExp;
            }
        }

        Debug.Assert(res.Cast<float>().All(x => !float.IsNaN(x)), "There should be no NaN values");

        return res;
    }

    /// <summary>
    /// Applies the Softplus activation function to each element of the specified two-dimensional array.
    /// </summary>
    /// <remarks>The Softplus function is defined as <c>log(1 + exp(x))</c> and is commonly used as a smooth
    /// approximation to the ReLU activation in machine learning applications. The returned array is a new instance; the
    /// input array is not modified.</remarks>
    /// <param name="source">A two-dimensional array of single-precision floating-point values to which the Softplus function will be
    /// applied.</param>
    /// <returns>A two-dimensional array of the same dimensions as <paramref name="source"/>, where each element is the result of
    /// applying the Softplus function to the corresponding element in <paramref name="source"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Softplus(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = MathF.Log(1 + MathF.Exp(source[i, j]));
            }
        }
        return res;
    }

    /// <summary>
    /// Splits the source into two sets of rows based on the specified ratio. 
    /// </summary>
    /// <param name="source">The two-dimensional array to split.</param>
    /// <param name="ratio">The ratio for splitting the rows.</param>
    /// <returns>A tuple containing the two sets of rows.</returns>
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
    /// Standardizes the source in-place so that each column (or a specified range of columns) has a mean of 0 and a standard deviation of 1 by applying the <c>(x - mean) / stdDev</c> transformation.
    /// </summary>
    /// <remarks>
    /// Standard deviation is calculated using the formula: <c>sqrt(sum((x - mean)^2) / N)</c>, where N is the number of rows.
    /// </remarks>
    /// <param name="source">The source to standardize.</param>
    /// <param name="columnRange">
    /// Optional. The range of columns to standardize. If null, all columns are standardized.
    /// </param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Standardize(this float[,] source, Range? columnRange = null)
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

    /// <summary>
    /// Standardizes the source in-place so that each column (or a specified range of columns) has mean 0 and standard deviation 1, computed in a single pass.
    /// </summary>
    /// <param name="source">The source to standardize.</param>
    /// <param name="columnRange">
    /// Optional. The range of columns to standardize. If null, all columns are standardized.
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
            float variance = (sumOfSquares / rows) - (mean * mean);
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

    /// <summary>
    /// Calculates the standard deviation for all elements of a two-dimensional array.
    /// </summary>
    /// <remarks>
    /// Standard deviation is calculated using the formula: <c>sqrt(sum((x - mean)^2) / N)</c>, where N is the number of all elements in the array.
    /// </remarks>
    /// <param name="source">The array whose standard deviation will be computed.</param>
    /// <returns>The standard deviation of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float StdDev(this float[,] source)
    {
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
    /// Calculates the standard deviation for all elements of a four-dimensional array.
    /// </summary>
    /// <remarks>
    /// Standard deviation is calculated using the formula: <c>sqrt(sum((x - mean)^2) / N)</c>, where N is the number of all elements in the array.
    /// </remarks>
    /// <param name="source">The array whose standard deviation will be computed.</param>
    /// <returns>The standard deviation of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float StdDev(this float[,,,] source)
    {
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
    /// Calculates the sum of all elements in the four-dimensional array.
    /// </summary>
    /// <param name="source">The array to sum.</param>
    /// <returns>The sum of all elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(this float[,,,] source)
    {
        // Sum over all elements.
        float sum = 0;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);
        int depth = source.GetLength(2);
        int channels = source.GetLength(3);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                for (int k = 0; k < depth; k++)
                {
                    for (int l = 0; l < channels; l++)
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
    /// Applies the hyperbolic tangent function element-wise to the source.
    /// </summary>
    /// <returns>A new source with the hyperbolic tangent applied element-wise.</returns>
    /// <param name="source">The two-dimensional array to transform.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Tanh(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = MathF.Tanh(source[i, j]);
            }
        }

        return res;
    }

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise to the source.
    /// </summary>
    /// <returns>A new source with the hyperbolic tangent applied element-wise.</returns>
    /// <param name="source">The four-dimensional array to transform.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Tanh(this float[,,,] source)
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
                        res[i, j, k, l] = MathF.Tanh(source[i, j, k, l]);
                    }
                }
            }
        }

        return res;
    }

    /// <summary>
    /// Transposes the source by swapping its rows and columns.
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

    ///// <summary>
    ///// Gets a row from the source.
    ///// </summary>
    ///// <param name="row">The index of the row to retrieve.</param>
    ///// <returns>A new <see cref="Matrix"/> object representing the specified row.</returns>
    ///// <remarks>
    ///// The returned row is a new instance of the <see cref="Matrix"/> class and has the same number of dim2 as the original source.
    ///// </remarks>
    //[MethodImpl(MethodImplOptions.AggressiveInlining)]
    //public static float[] GetRow(this float[,] source, int row)
    //{
    //    int dim2 = source.GetLength(1);

    //    // Create an array to store the row.
    //    float[] res = new float[dim2];
    //    for (int i = 0; i < dim2; i++)
    //    {
    //        // Access each element in the specified row.
    //        res[i] = source[row, i];
    //    }

    //    return res;
    //}
}