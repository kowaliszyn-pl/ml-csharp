// Neural Networks in C♯
// File name: ArraySlicingRoutines.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

/// <summary>
/// Provides array slicing and row/column manipulation routines for supported multidimensional array types.
/// </summary>
/// <remarks>Includes extension methods for selecting rows, columns, and ranges across dimensions, splitting row
/// sets by ratio, and writing row values back into arrays. Most slicing operations return new arrays rather than
/// views.</remarks>
public static class ArraySlicingRoutines
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T SliceDim0<T>(T source, int startInclusive, int endExclusive)
        where T : notnull
    {
        Range range = startInclusive..endExclusive;

        return source switch
        {
            float[,] a => (T)(object)a.GetRows(range),
            float[,,] a => (T)(object)a.GetRows(range),
            float[,,,] a => (T)(object)a.GetRows(range),
            int[,] a => (T)(object)a.GetRows(range),
            _ => throw new NotSupportedException($"Slicing not supported for array type: {source.GetType().Name}.")
        };
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
    /// Gets a submatrix containing the specified range of columns from the current source. The shape is [rows, range].
    /// </summary>
    /// <param name="source">The array to slice.</param>
    /// <param name="range">The range specifying the subset of columns.</param>
    /// <returns>A new array with [rows, selectedColumns].</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] GetColumns(this float[,] source, Range range)
    {
        (int offset, int columns) = range.GetOffsetAndLength(source.GetLength(1));

        int rows = source.GetLength(0);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
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
    /// <returns>A new array representing the specified row.</returns>
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
    /// <returns>A new three-dimensional array with shape [dim1, dim2, dim3] containing the selected row.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,] GetRow(this float[,,,] source, int row)
    {
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);
        int dim3 = source.GetLength(3);

        // Create an array to store the row.
        float[,,] res = new float[dim1, dim2, dim3];
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
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
    /// <returns>A new array representing the submatrix.</returns>
    /// <remarks>
    /// The returned columns are the same as in the original source.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] GetRows(this float[,] source, Range range)
    {
        (int offset, int rows) = range.GetOffsetAndLength(source.GetLength(0));

        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = source[i + offset, j];
            }
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int[,] GetRows(this int[,] source, Range range)
    {
        (int offset, int rows) = range.GetOffsetAndLength(source.GetLength(0));

        int columns = source.GetLength(1);
        int[,] res = new int[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = source[i + offset, j];
            }
        }

        return res;
    }

    /// <summary>
    /// Gets a contiguous range of rows from a three-dimensional array.
    /// </summary>
    /// <param name="source">The three-dimensional array to slice.</param>
    /// <param name="range">The range applied to rows.</param>
    /// <returns>A new three-dimensional array containing the selected rows.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,] GetRows(this float[,,] source, Range range)
    {
        (int offset, int dim0) = range.GetOffsetAndLength(source.GetLength(0));

        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);

        float[,,] res = new float[dim0, dim1, dim2];

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    res[i, j, k] = source[i + offset, j, k];
                }
            }
        }

        return res;

    }

    /// <summary>
    /// Gets a contiguous range of rows from a four-dimensional array.
    /// </summary>
    /// <param name="source">The four-dimensional array to slice.</param>
    /// <param name="range">The range applied to rows.</param>
    /// <returns>A new four-dimensional array containing the selected rows.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] GetRows(this float[,,,] source, Range range)
    {
        (int offset, int dim0) = range.GetOffsetAndLength(source.GetLength(0));

        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);
        int dim3 = source.GetLength(3);

        float[,,,] res = new float[dim0, dim1, dim2, dim3];

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    for (int l = 0; l < dim3; l++)
                    {
                        res[i, j, k, l] = source[i + offset, j, k, l];
                    }
                }
            }
        }

        return res;

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
        int splitIndex = (int)(source.GetLength(0) * ratio);
        return (source.GetRows(..splitIndex), source.GetRows(splitIndex..));

        //int rows = source.GetLength(0);
        //int columns = source.GetLength(1);
        //int splitIndex = (int)(rows * ratio);
        //float[,] set1 = new float[splitIndex, columns];
        //float[,] set2 = new float[rows - splitIndex, columns];
        //for (int i = 0; i < rows; i++)
        //{
        //    for (int j = 0; j < columns; j++)
        //    {
        //        if (i < splitIndex)
        //        {
        //            set1[i, j] = source[i, j];
        //        }
        //        else
        //        {
        //            set2[i - splitIndex, j] = source[i, j];
        //        }
        //    }
        //}
        //return (set1, set2);
    }

    /// <summary>
    /// Sets the values of a specific row in the source.
    /// </summary>
    /// <param name="source">The two-dimensional array to modify.</param>
    /// <param name="rowIndex">The index of the row to set.</param>
    /// <param name="row">The array containing the values to set. Length must equal the number of columns.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SetRow(this float[,] source, int rowIndex, float[] row)
    {
        Debug.Assert(rowIndex >= 0 && rowIndex < source.GetLength(0), "Row index out of bounds.");

        int columns = source.GetLength(1);
        Debug.Assert(row.Length == columns, "Row length does not match the number of columns.");
        
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
    /// <param name="row">A three-dimensional array with shape [dim1, dim2, dim3] providing values.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SetRow(this float[,,,] source, int rowIndex, float[,,] row)
    {
        Debug.Assert(rowIndex >= 0 && rowIndex < source.GetLength(0), "Row index out of bounds.");

        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);
        int dim3 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    source[rowIndex, i, j, k] = row[i, j, k];
                }
            }
        }
    }
}
