// Neural Networks in C♯
// File name: ArrayRoutines.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Core;

/// <summary>
/// Provides extension methods for comparing the shape of single- and multi-dimensional <see cref="float"/> arrays.
/// </summary>
/// <remarks>Includes overloads that verify dimension lengths match for array ranks one through four. Useful for
/// validating array compatibility before element-wise operations.</remarks>
public static class ArrayRoutines
{
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
    /// <returns>True if both arrays have equal rows and columns; otherwise false.</returns>
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
}