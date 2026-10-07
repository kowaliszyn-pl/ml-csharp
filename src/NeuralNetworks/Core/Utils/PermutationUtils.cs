// Neural Networks in C♯
// File name: PermutationUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Utils;

public static class PermutationUtils
{
    public enum PermuteMethod
    {
        Indices, FisherYates
    }

    public enum CopyMethod
    {
        CopyValues, SetRow
    }

    #region Arrays float[,]

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlace(this float[,] source, Random? random = null, PermuteMethod permuteMethod = PermuteMethod.Indices, CopyMethod copyMethod = CopyMethod.CopyValues) => PermuteInPlaceInternal(source, null, random, permuteMethod, copyMethod);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Permute(this float[,] source, Random? random = null, PermuteMethod permuteMethod = PermuteMethod.Indices, CopyMethod copyMethod = CopyMethod.CopyValues)
    {
        float[,] res = (float[,])source.Clone();
        PermuteInPlaceInternal(res, null, random, permuteMethod, copyMethod);
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceTogetherWith(this float[,] source, float[,] y, Random? random = null, PermuteMethod permuteMethod = PermuteMethod.Indices, CopyMethod copyMethod = CopyMethod.CopyValues) => PermuteInPlaceInternal(source, y, random, permuteMethod, copyMethod);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float[,] xPermuted, float[,] yPermuted) Permute(float[,] x, float[,] y, Random? random = null, PermuteMethod permuteMethod = PermuteMethod.Indices, CopyMethod copyMethod = CopyMethod.CopyValues)
    {
        float[,] xRes = (float[,])x.Clone();
        float[,] yRes = (float[,])y.Clone();
        PermuteInPlaceInternal(xRes, yRes, random, permuteMethod, copyMethod);
        return (xRes, yRes);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteTogetherWith(float[,] x, float[,] y, Random? random = null, PermuteMethod permuteMethod = PermuteMethod.Indices, CopyMethod copyMethod = CopyMethod.CopyValues) => PermuteInPlaceInternal(x, y, random, permuteMethod, copyMethod);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PermuteInPlaceInternal(float[,] x, float[,]? y, Random? random, PermuteMethod permuteMethod, CopyMethod copyMethod)
    {
        Debug.Assert(y == null || x.GetLength(0) == y.GetLength(0), "Both matrices must have the same number of rows to permute them together.");

        random ??= new();

        int rows = x.GetLength(0);
        int xColumns = x.GetLength(1);
        int yColumns = y?.GetLength(1) ?? 0;

        if (permuteMethod == PermuteMethod.Indices)
        {
            int[] indices = [.. Enumerable.Range(0, rows).OrderBy(i => random.Next())];
            float[,] xCopy = (float[,])x.Clone();
            float[,]? yCopy = y == null ? null : (float[,])y.Clone();

            if (copyMethod == CopyMethod.CopyValues)
            {
                for (int i = 0; i < rows; i++)
                {
                    int fromRow = indices[i];
                    for (int j = 0; j < xColumns; j++)
                    {
                        x[i, j] = xCopy[fromRow, j];
                    }

                    if (yCopy != null)
                        for (int j = 0; j < yColumns; j++)
                        {
                            y![i, j] = yCopy[fromRow, j];
                        }
                }
            }
            else if (copyMethod == CopyMethod.SetRow)
            {
                for (int i = 0; i < rows; i++)
                {
                    x.SetRow(i, xCopy.GetRow(indices[i]));
                    if (yCopy != null)
                    {
                        y!.SetRow(i, yCopy.GetRow(indices[i]));
                    }
                }
            }
        }
        else if (permuteMethod == PermuteMethod.FisherYates)
        {
            if (copyMethod == CopyMethod.CopyValues)
            {
                for (int i = rows - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    if (i != j)
                    {
                        // Swap rows in x
                        for (int col = 0; col < xColumns; col++)
                        {
                            (x[i, col], x[j, col]) = (x[j, col], x[i, col]);
                        }
                        // Swap rows in y if applicable
                        if (y != null)
                        {
                            for (int col = 0; col < yColumns; col++)
                            {
                                (y[i, col], y[j, col]) = (y[j, col], y[i, col]);
                            }
                        }
                    }
                }
            }
            else if (copyMethod == CopyMethod.SetRow)
            {
                for (int i = rows - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    if (i != j)
                    {
                        // Swap rows in x
                        float[] tempX = x.GetRow(i);
                        x.SetRow(i, x.GetRow(j));
                        x.SetRow(j, tempX);
                        // Swap rows in y if applicable
                        if (y != null)
                        {
                            float[] tempY = y.GetRow(i);
                            y.SetRow(i, y.GetRow(j));
                            y.SetRow(j, tempY);
                        }
                    }
                }
            }
        }
    }

    #endregion
}
