// Neural Networks in C♯
// File name: ReshapeRoutines.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

/// <summary>
/// Provides reshape routines for converting between 4D and 2D <see cref="float"/> arrays by flattening and
/// restoring trailing dimensions.
/// </summary>
/// <remarks>Preserves element order by mapping each 4D slice <c>[dim1, dim2, dim3]</c> to a contiguous 2D row
/// segment and supports round-trip conversion when the 2D column count equals <c>dim1 * dim2 * dim3</c>.</remarks>
public static class ReshapeRoutines
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] ReshapeTo2D(this float[,,,] data)
    {
        int dim0 = data.GetLength(0);
        int dim1 = data.GetLength(1);
        int dim2 = data.GetLength(2);
        int dim3 = data.GetLength(3);

        float[,] res = new float[dim0, dim1 * dim2 * dim3];
        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                int jOffset = j * dim2 * dim3;
                for (int k = 0; k < dim2; k++)
                {
                    int kOffset = jOffset + k * dim3;
                    for (int l = 0; l < dim3; l++)
                    {
                        res[i, kOffset + l] = data[i, j, k, l];
                    }
                }
            }
        }
        return res;
    }

    // TODO: round-trip tests 4D -> 2D -> 4D preserving values for a few non-trivial shapes

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] ReshapeTo4D(this float[,] data, int dim1, int dim2, int dim3)
    {
        int rows = data.GetLength(0);
        int columns = data.GetLength(1);

        Debug.Assert(columns == dim1 * dim2 * dim3);

        float[,,,] res = new float[rows, dim1, dim2, dim3];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                int kOffset = j * dim2 * dim3;
                for (int k = 0; k < dim2; k++)
                {
                    int lOffset = kOffset + k * dim3;
                    for (int l = 0; l < dim3; l++)
                    {
                        res[i, j, k, l] = data[i, lOffset + l];
                    }
                }
            }
        }
        return res;
    }

}
