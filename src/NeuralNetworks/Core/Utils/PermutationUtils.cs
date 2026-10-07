// Neural Networks in C♯
// File name: PermutationUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Utils;

public static class PermutationUtils
{
    #region Arrays float[,], float[,]

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlace(this float[,] source, Random? random = null)
        => PermuteInPlaceInternal(source, null, random);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Permute(this float[,] source, Random? random = null)
    {
        float[,] res = (float[,])source.Clone();
        PermuteInPlaceInternal(res, null, random);
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this float[,] source, float[,] y, Random? random = null)
        => PermuteInPlaceInternal(source, y, random);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float[,] xPermuted, float[,] yPermuted) Permute(float[,] x, float[,] y, Random? random = null)
    {
        float[,] xRes = (float[,])x.Clone();
        float[,] yRes = (float[,])y.Clone();
        PermuteInPlaceInternal(xRes, yRes, random);
        return (xRes, yRes);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteTogetherWith(float[,] x, float[,] y, Random? random = null)
        => PermuteInPlaceInternal(x, y, random);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PermuteInPlaceInternal(float[,] x, float[,]? y, Random? random)
    {
        Debug.Assert(y == null || x.GetLength(0) == y.GetLength(0), "Both matrices must have the same number of rows to permute them together.");

        random ??= new();

        int xDim1 = x.GetLength(0);
        int xDim2 = x.GetLength(1);
        int yDim2 = y?.GetLength(1) ?? 0;

        for (int i1 = xDim1 - 1; i1 > 0; i1--)
        {
            int i2 = random.Next(i1 + 1);
            if (i1 != i2)
            {
                // Swap rows in x
                SwapMatrixRows(x, xDim2, i1, i2);

                // Swap rows in y if applicable
                if (y != null)
                {
                    SwapMatrixRows(y, yDim2, i1, i2);
                }
            }
        }
    }

    #endregion

    #region Arrays float[,,], float[,]

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this float[,,] source, float[,] y, Random? random = null)
        => PermuteInPlaceInternal(source, y, random);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PermuteInPlaceInternal(float[,,] x, float[,]? y, Random? random)
    {
        Debug.Assert(y == null || x.GetLength(0) == y.GetLength(0), "Both matrices must have the same number of rows to permute them together.");

        random ??= new();

        int xDim1 = x.GetLength(0);
        int xDim2 = x.GetLength(1);
        int xDim3 = x.GetLength(2);
        int yDim2 = y?.GetLength(1) ?? 0;

        for (int i1 = xDim1 - 1; i1 > 0; i1--)
        {
            int i2 = random.Next(i1 + 1);
            if (i1 != i2)
            {
                // Swap rows in x
                SwapMatrixRows(x, xDim2, xDim3, i1, i2);

                // Swap rows in y if applicable
                if (y != null)
                {
                    SwapMatrixRows(y, yDim2, i1, i2);
                }
            }
        }
    }

    #endregion

    #region Arrays float[,,,], float[,]

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this float[,,,] source, float[,] y, Random? random = null)
        => PermuteInPlaceInternal(source, y, random);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PermuteInPlaceInternal(float[,,,] x, float[,]? y, Random? random)
    {
        Debug.Assert(y == null || x.GetLength(0) == y.GetLength(0), "Both matrices must have the same number of rows to permute them together.");

        random ??= new();

        int xDim1 = x.GetLength(0);
        int xDim2 = x.GetLength(1);
        int xDim3 = x.GetLength(2);
        int xDim4 = x.GetLength(3);
        int yDim2 = y?.GetLength(1) ?? 0;

        for (int i1 = xDim1 - 1; i1 > 0; i1--)
        {
            int i2 = random.Next(i1 + 1);
            if (i1 != i2)
            {
                // Swap rows in x
                SwapMatrixRows(x, xDim2, xDim3, xDim4, i1, i2);

                // Swap rows in y if applicable
                if (y != null)
                {
                    SwapMatrixRows(y, yDim2, i1, i2);
                }
            }
        }
    }

    #endregion

    #region Arrays float[,,,], float[,,,]

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this float[,,,] source, float[,,,] y, Random? random = null)
        => PermuteInPlaceInternal(source, y, random);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PermuteInPlaceInternal(float[,,,] x, float[,,,]? y, Random? random)
    {
        Debug.Assert(y == null || x.GetLength(0) == y.GetLength(0), "Both matrices must have the same number of rows to permute them together.");

        random ??= new();

        int xDim1 = x.GetLength(0);
        int xDim2 = x.GetLength(1);
        int xDim3 = x.GetLength(2);
        int xDim4 = x.GetLength(3);
        int yDim2 = y?.GetLength(1) ?? 0;
        int yDim3 = y?.GetLength(2) ?? 0;
        int yDim4 = y?.GetLength(3) ?? 0;

        for (int i1 = xDim1 - 1; i1 > 0; i1--)
        {
            int i2 = random.Next(i1 + 1);
            if (i1 != i2)
            {
                // Swap rows in x
                SwapMatrixRows(x, xDim2, xDim3, xDim4, i1, i2);

                // Swap rows in y if applicable
                if (y != null)
                {
                    SwapMatrixRows(y, yDim2, yDim3, yDim4, i1, i2);
                }
            }
        }
    }

    #endregion

    #region Arrays int[,], float[,]

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this int[,] source, float[,] y, Random? random = null)
        => PermuteInPlaceInternal(source, y, random);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PermuteInPlaceInternal(int[,] x, float[,]? y, Random? random)
    {
        Debug.Assert(y == null || x.GetLength(0) == y.GetLength(0), "Both matrices must have the same number of rows to permute them together.");

        random ??= new();

        int xDim1 = x.GetLength(0);
        int xDim2 = x.GetLength(1);
        int yDim2 = y?.GetLength(1) ?? 0;

        for (int i1 = xDim1 - 1; i1 > 0; i1--)
        {
            int i2 = random.Next(i1 + 1);
            if (i1 != i2)
            {
                // Swap rows in x
                SwapMatrixRows(x, xDim2, i1, i2);

                // Swap rows in y if applicable
                if (y != null)
                {
                    SwapMatrixRows(y, yDim2, i1, i2);
                }
            }
        }
    }

    #endregion

    #region Swap

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SwapMatrixRows(float[,] x, int xDim2, int i1, int i2)
    {
        for (int j = 0; j < xDim2; j++)
        {
            (x[i1, j], x[i2, j]) = (x[i2, j], x[i1, j]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SwapMatrixRows(float[,,] x, int xDim2, int xDim3, int i1, int i2)
    {
        for (int j = 0; j < xDim2; j++)
        {
            for (int k = 0; k < xDim3; k++)
            {
                (x[i1, j, k], x[i2, j, k]) = (x[i2, j, k], x[i1, j, k]);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SwapMatrixRows(float[,,,] x, int xDim2, int xDim3, int xDim4, int i1, int i2)
    {
        for (int j = 0; j < xDim2; j++)
        {
            for (int k = 0; k < xDim3; k++)
            {
                for (int l = 0; l < xDim4; l++)
                {
                    (x[i1, j, k, l], x[i2, j, k, l]) = (x[i2, j, k, l], x[i1, j, k, l]);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SwapMatrixRows(int[,] x, int xDim2, int i1, int i2)
    {
        for (int j = 0; j < xDim2; j++)
        {
            (x[i1, j], x[i2, j]) = (x[i2, j], x[i1, j]);
        }
    }

    #endregion
}
