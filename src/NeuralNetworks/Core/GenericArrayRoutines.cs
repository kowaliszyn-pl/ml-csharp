// Neural Networks in C♯
// File name: GenericArrayRoutines.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Runtime.InteropServices;

namespace NeuralNetworks.Core;

/// <summary>
/// Provides utility methods for working with single- and multi-dimensional <see cref="float"/> arrays, including span
/// conversion, shape validation, and row-count retrieval.
/// </summary>
/// <remarks>Span conversion supports <c>float[]</c>, <c>float[,]</c>, <c>float[,,]</c>, and <c>float[,,,]</c>.
/// Multi-dimensional arrays are projected as contiguous spans over their underlying storage.</remarks>
public static class GenericArrayRoutines
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

}