// Neural Networks in C♯
// File name: ArrayExtensions.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NeuralNetworks.Core.Span;

public static class ArrayExtensions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Flatten(this float[,,,] source)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        Debug.Assert(dim1 > 0 && dim2 > 0 && dim3 > 0 && dim4 > 0, "All dimensions must be greater than zero.");

        float[,] res = new float[dim1, dim2 * dim3 * dim4];

        ref float sourceRef = ref source[0, 0, 0, 0];
        ref float resRef = ref res[0, 0];

        ReadOnlySpan<float> sourceSpan = MemoryMarshal.CreateReadOnlySpan(ref sourceRef, source.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref resRef, res.Length);

        sourceSpan.CopyTo(resSpan);

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] LeakyReLU(this float[,,,] source, float alpha = 0.01f, float beta = 1f)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        ref float sourceRef = ref source[0, 0, 0, 0];
        ref float resRef = ref res[0, 0, 0, 0];

        ReadOnlySpan<float> sourceSpan = MemoryMarshal.CreateReadOnlySpan(ref sourceRef, source.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref resRef, res.Length);
        
        for(int i = 0; i < sourceSpan.Length; i++)
        {
            float value = sourceSpan[i];
            resSpan[i] = value >= 0 ? value * beta : value * alpha;
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

        Debug.Assert(dim1 > 0 && dim2 > 0 && dim3 > 0 && dim4 > 0, "All dimensions must be greater than zero.");

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        ref float sourceRef = ref source[0, 0, 0, 0];
        ref float resRef = ref res[0, 0, 0, 0];

        ReadOnlySpan<float> sourceSpan = MemoryMarshal.CreateReadOnlySpan(ref sourceRef, source.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref resRef, res.Length);

        for (int i = 0; i < resSpan.Length; i++)
        {
            resSpan[i] = MathF.Tanh(sourceSpan[i]);
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Unflatten(this float[,] source, float[,,,] targetSize)
    {
        int dim1 = targetSize.GetLength(0);
        int dim2 = targetSize.GetLength(1);
        int dim3 = targetSize.GetLength(2);
        int dim4 = targetSize.GetLength(3);

        Debug.Assert(dim1 > 0 && dim2 > 0 && dim3 > 0 && dim4 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(source.GetLength(0) == dim1 && source.GetLength(1) == dim2 * dim3 * dim4, "Source shape does not match target size for unflattening.");

        float[,,,] res = new float[dim1, dim2, dim3, dim4];
        ref float sourceRef = ref source[0, 0];
        ref float resRef = ref res[0, 0, 0, 0];

        ReadOnlySpan<float> sourceSpan = MemoryMarshal.CreateReadOnlySpan(ref sourceRef, source.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref resRef, res.Length);

        sourceSpan.CopyTo(resSpan);

        return res;
    }

}
