// Neural Networks in C♯
// File name: ArrayExtensionsSpan.cs
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
        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        int dim2 = source.GetLength(2);
        int dim3 = source.GetLength(3);

        Debug.Assert(dim0 > 0 && dim1 > 0 && dim2 > 0 && dim3 > 0, "All dimensions must be greater than zero.");

        float[,] res = new float[dim0, dim1 * dim2 * dim3];
        ref float sourceRef = ref source[0, 0, 0, 0];
        ref float resRef = ref res[0, 0];
        ReadOnlySpan<float> sourceSpan = MemoryMarshal.CreateReadOnlySpan(ref sourceRef, source.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref resRef, res.Length);
        for (int b = 0; b < dim0; b++)
        {
            for (int c = 0; c < dim1; c++)
            {
                for (int h = 0; h < dim2; h++)
                {
                    for (int w = 0; w < dim3; w++)
                    {
                        int index = c * dim2 * dim3 + h * dim3 + w;
                        resSpan[b * (dim1 * dim2 * dim3) + index] =
                            sourceSpan[b * (dim1 * dim2 * dim3) + c * (dim2 * dim3) + h * dim3 + w];
                    }
                }
            }
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] MultiplyByTanhDerivative(this float[,,,] outputGradient, float[,,,] output)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Tanh function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Tanh function tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) is 1 - tanh(x)^2.
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * (1 - tanh(x)^2).
        // The elementwise multiplication of the output gradient and the derivative of the Tanh function is returned as the input gradient.
        // tanh(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient

        int d0 = outputGradient.GetLength(0);
        int d1 = outputGradient.GetLength(1);
        int d2 = outputGradient.GetLength(2);
        int d3 = outputGradient.GetLength(3);

        Debug.Assert(d0 > 0 && d1 > 0 && d2 > 0 && d3 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(output.GetLength(0) != d0 && output.GetLength(1) != d1 && output.GetLength(2) != d2 && output.GetLength(3) != d3, "Shapes of outputGradient and output must match for elementwise operations.");

        float[,,,] result = new float[d0, d1, d2, d3];

        ref float ogRef = ref outputGradient[0, 0, 0, 0];
        ref float outRef = ref output[0, 0, 0, 0];
        ref float resRef = ref result[0, 0, 0, 0];

        ReadOnlySpan<float> ogSpan = MemoryMarshal.CreateReadOnlySpan(ref ogRef, outputGradient.Length);
        ReadOnlySpan<float> outSpan = MemoryMarshal.CreateReadOnlySpan(ref outRef, output.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref resRef, result.Length);

        for (int i = 0; i < resSpan.Length; i++)
        {
            float y = outSpan[i];
            float dy = ogSpan[i];
            resSpan[i] = dy * (1f - (y * y));
        }

        return result;
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

        /*
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        res[i, j, k, l] = MathF.Tanh(source[i, j, k, l]);
                    }
                }
            }
        }*/

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Unflatten(this float[,] source, float[,,,] targetSize)
    {
        int dim0 = targetSize.GetLength(0);
        int dim1 = targetSize.GetLength(1);
        int dim2 = targetSize.GetLength(2);
        int dim3 = targetSize.GetLength(3);

        Debug.Assert(dim0 > 0 && dim1 > 0 && dim2 > 0 && dim3 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(source.GetLength(0) == dim0 && source.GetLength(1) == dim1 * dim2 * dim3, "Source shape does not match target size for unflattening.");

        float[,,,] res = new float[dim0, dim1, dim2, dim3];
        ref float sourceRef = ref source[0, 0];
        ref float resRef = ref res[0, 0, 0, 0];
        ReadOnlySpan<float> sourceSpan = MemoryMarshal.CreateReadOnlySpan(ref sourceRef, source.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref resRef, res.Length);
        for (int b = 0; b < dim0; b++)
        {
            for (int c = 0; c < dim1; c++)
            {
                for (int h = 0; h < dim2; h++)
                {
                    for (int w = 0; w < dim3; w++)
                    {
                        int index = c * dim2 * dim3 + h * dim3 + w;
                        resSpan[b * (dim1 * dim2 * dim3) + c * (dim2 * dim3) + h * dim3 + w] =
                            sourceSpan[b * (dim1 * dim2 * dim3) + index];
                    }
                }
            }
        }
        return res;

    }
}
