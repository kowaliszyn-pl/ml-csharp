// Neural Networks in C♯
// File name: ArrayExtensions.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;

public static class ArrayExtensions
{
    /// <summary>
    /// Adds a scalar value to each element of the matrix.
    /// </summary>
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
    /// Calculates the mean of all elements in the matrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Mean(this float[,] source)
        => source.Sum() / source.Length;

    /// <summary>
    /// Multiplies each element of the matrix by a scalar value.
    /// </summary>
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
    /// Multiplies the current matrix with another matrix using the dot product.
    /// </summary>
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
    /// Raises each element of the matrix to the specified power.
    /// </summary>
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
    /// Subtracts the elements of the specified matrix from the current matrix.
    /// </summary>
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
    /// Calculates the sum of all elements in the matrix.
    /// </summary>
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
    /// Transposes the matrix by swapping its rows and columns.
    /// </summary>
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

    /// <summary>
    /// Gets a row from the matrix.
    /// </summary>
    /// <param name="row">The index of the row to retrieve.</param>
    /// <returns>A new <see cref="Matrix"/> object representing the specified row.</returns>
    /// <remarks>
    /// The returned row is a new instance of the <see cref="Matrix"/> class and has the same number of columns as the original matrix.
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
}