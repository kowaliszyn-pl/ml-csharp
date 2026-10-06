// Neural Networks in C♯
// File name: GenericArrayUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Runtime.InteropServices;

namespace NeuralNetworks.Core;

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

    public static void PermuteData<TX, TY>(TX matrix1, TY matrix2, Random random)
        where TX : notnull
        where TY : notnull
    {
        // TODO: let's use Span<T> (and Memory<T>?) to make it more efficient and avoid copying data when possible.
        switch (matrix1, matrix2)
        {
            case (float[,] x2, float[,] y2):
                x2.PermuteInPlaceTogetherWith(y2, random);
                return;

            case (float[,,] x3, float[,] y2):
                x3.PermuteInPlaceTogetherWith(y2, random);
                return;

            case (float[,,,] x4, float[,] y2):
                x4.PermuteInPlaceTogetherWith(y2, random);
                return;

            case (float[,,,] x4, float[,,,] y2):
                x4.PermuteInPlaceTogetherWith(y2, random);
                return;

            case (int[,] x5, float[,] y2): // embedding indices and labels
                x5.PermuteInPlaceTogetherWith(y2, random);
                return;

            default:
                throw new NotSupportedException($"Unsupported permutation pair: x={matrix1.GetType().Name}, y={matrix2.GetType().Name}. Please override this method in the Trainer subclass or add here a permutation method for these data types.");
        }
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
