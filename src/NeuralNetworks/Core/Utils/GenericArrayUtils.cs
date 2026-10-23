// Neural Networks in C♯
// File name: GenericArrayUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Runtime.InteropServices;

namespace NeuralNetworks.Core.Utils;

/// <summary>
/// Provides helper methods for working with supported numeric arrays of varying ranks, including span conversion, shape
/// validation, row-count retrieval, coordinated permutation, and slicing along the first dimension.
/// </summary>
/// <remarks>Designed for array-based tensor-like data and supports specific combinations such as <c>float[]</c>,
/// <c>float[,]</c>, <c>float[,,]</c>, <c>float[,,,]</c>, and related paired types used for synchronized
/// operations.</remarks>
public static class GenericArrayUtils
{
    public static Span<float> ConvertToSpan<T>(T matrix)
    {
        return matrix switch
        {
            float[] arr1D => arr1D.AsSpan(),
            float[,] arr2D => MemoryMarshal.CreateSpan(ref arr2D[0, 0], arr2D.Length),
            float[,,] arr3D => MemoryMarshal.CreateSpan(ref arr3D[0, 0, 0], arr3D.Length),
            float[,,,] arr4D => MemoryMarshal.CreateSpan(ref arr4D[0, 0, 0, 0], arr4D.Length),
            _ => throw new ArgumentException("Array must be one of the following types: float[], float[,], float[,,], float[,,,].")
        };
    }

    public static ReadOnlySpan<float> ConvertToReadOnlySpan<T>(T matrix)
    {
        return matrix switch
        {
            float[] arr1D => arr1D.AsSpan(),
            float[,] arr2D => MemoryMarshal.CreateReadOnlySpan(ref arr2D[0, 0], arr2D.Length),
            float[,,] arr3D => MemoryMarshal.CreateReadOnlySpan(ref arr3D[0, 0, 0], arr3D.Length),
            float[,,,] arr4D => MemoryMarshal.CreateReadOnlySpan(ref arr4D[0, 0, 0, 0], arr4D.Length),
            _ => throw new ArgumentException("Array must be one of the following types: float[], float[,], float[,,], float[,,,].")
        };
    }

    public static void EnsureSameShape<T>(T? matrix1, T? matrix2)
    {
        if (matrix1 is null || matrix2 is null)
            throw new ArgumentException("Matrix is null.");

        // Check if input and inputGradient are both arrays and have the same shape.
        if (matrix1 is Array matrix1Array && matrix2 is Array matrix2Array)
        {
            if (matrix1Array.Rank != matrix2Array.Rank)
                throw new InvalidOperationException($"Matrices must have the same number of dimensions. Got {matrix1Array.Rank} and {matrix2Array.Rank}.");
            for (int i = 0; i < matrix1Array.Rank; i++)
            {
                if (matrix1Array.GetLength(i) != matrix2Array.GetLength(i))
                    throw new InvalidOperationException($"Matrices must have the same shape. Dimension {i} has length {matrix1Array.GetLength(i)} and {matrix2Array.GetLength(i)}.");
            }
        }
    }

    public static int GetRowCount<T>(T matrix)
    {
        if (matrix is Array array)
            return array.GetLength(0);

        throw new NotSupportedException();
    }

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
}
