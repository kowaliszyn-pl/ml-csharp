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
        Debug.Assert(output.GetLength(0) == d0 && output.GetLength(1) == d1 && output.GetLength(2) == d2 && output.GetLength(3) == d3, "Shapes of outputGradient and output must match for elementwise operations.");

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

    /// <summary>
    /// 2D convolution forward pass on NHWC-like 4D tensors:
    /// Input: [batch, inChannels, inHeight, inWidth]
    /// Weights: [inChannels, outChannels, kernelHeight, kernelWidth]
    /// Output: [batch, outChannels, outHeight, outWidth]
    /// Padding is symmetric and computed as kernelSize / 2 (same padding) if not specified.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Convolve2DForward(this float[,,,] input, float[,,,] weights, int? padding = null)
    {
        int batchSize = input.GetLength(0);
        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int weightChannels = weights.GetLength(0);
        int outputChannels = weights.GetLength(1);
        int kernelHeight = weights.GetLength(2);
        int kernelWidth = weights.GetLength(3);

        Debug.Assert(weightChannels == inputChannels);
        Debug.Assert(kernelHeight == kernelWidth);

        int pad = padding ?? (kernelHeight / 2);

        int outputHeight = inputHeight - kernelHeight + 1 + 2 * pad;
        int outputWidth = inputWidth - kernelWidth + 1 + 2 * pad;

        float[,,,] output = new float[batchSize, outputChannels, outputHeight, outputWidth];

        ref float inputRef = ref input[0, 0, 0, 0];
        ref float weightsRef = ref weights[0, 0, 0, 0];
        ref float outputRef = ref output[0, 0, 0, 0];

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref inputRef, input.Length);
        ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weightsRef, weights.Length);
        Span<float> outputSpan = MemoryMarshal.CreateSpan(ref outputRef, output.Length);

        for (int b = 0; b < batchSize; b++)
        {
            for (int oc = 0; oc < outputChannels; oc++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        float sum = 0f;
                        for (int ic = 0; ic < inputChannels; ic++)
                        {
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int ih = oh + kh - pad;
                                    int iw = ow + kw - pad;
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                    {
                                        // sum += input[b, ic, ih, iw] * weights[ic, oc, kh, kw];
                                        sum += inputSpan[b * (inputChannels * inputHeight * inputWidth) + ic * (inputHeight * inputWidth) + ih * inputWidth + iw] *
                                               weightsSpan[ic * (outputChannels * kernelHeight * kernelWidth) + oc * (kernelHeight * kernelWidth) + kh * kernelWidth + kw];
                                    }
                                }
                            }
                        }
                        // output[b, oc, oh, ow] = sum;
                        outputSpan[b * (outputChannels * outputHeight * outputWidth) + oc * (outputHeight * outputWidth) + oh * outputWidth + ow] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Backward pass w.r.t. input for 2D convolution.
    /// inputGrad shape: [batch, inChannels, inHeight, inWidth]
    /// outputGrad shape: [batch, outChannels, outHeight, outWidth]
    /// weights shape: [inChannels, outChannels, kernelHeight, kernelWidth]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Convolve2DBackwardInput(this float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int? padding = null)
    {
        int batchSize = outputGradient.GetLength(0);
        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputChannels = outputGradient.GetLength(1);
        int kernelHeight = weights.GetLength(2);
        int kernelWidth = weights.GetLength(3);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        Debug.Assert(weights.GetLength(0) == inputChannels);
        Debug.Assert(weights.GetLength(1) == outputChannels);
        Debug.Assert(kernelHeight == kernelWidth);

        int pad = padding ?? (kernelHeight / 2);

        float[,,,] inputGradient = new float[batchSize, inputChannels, inputHeight, inputWidth];

        ref float weightsRef = ref weights[0, 0, 0, 0];
        ref float outputGradientRef = ref outputGradient[0, 0, 0, 0];
        ref float inputGradientRef = ref inputGradient[0, 0, 0, 0];

        ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weightsRef, weights.Length);
        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradientRef, outputGradient.Length);
        Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradientRef, inputGradient.Length);

        for (int b = 0; b < batchSize; b++)
        {
            for (int ic = 0; ic < inputChannels; ic++)
            {
                for (int ih = 0; ih < inputHeight; ih++)
                {
                    for (int iw = 0; iw < inputWidth; iw++)
                    {
                        float sum = 0f;
                        for (int oc = 0; oc < outputChannels; oc++)
                        {
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int oh = ih - kh + pad;
                                    int ow = iw - kw + pad;
                                    if (oh >= 0 && oh < outputGradientHeight && ow >= 0 && ow < outputGradientWidth)
                                    {
                                        // sum += outputGradient[b, oc, oh, ow] * weights[ic, oc, kh, kw];
                                        sum += outputGradientSpan[b * (outputChannels * outputGradientHeight * outputGradientWidth) + oc * (outputGradientHeight * outputGradientWidth) + oh * outputGradientWidth + ow] *
                                               weightsSpan[ic * (outputChannels * kernelHeight * kernelWidth) + oc * (kernelHeight * kernelWidth) + kh * kernelWidth + kw];
                                    }
                                }
                            }
                        }
                        // inputGradient[b, ic, ih, iw] = sum;
                        inputGradientSpan[b * (inputChannels * inputHeight * inputWidth) + ic * (inputHeight * inputWidth) + ih * inputWidth + iw] = sum;
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Backward pass w.r.t. weights (parameters) for 2D convolution.
    /// Returns gradient with shape [inChannels, outChannels, kernelHeight, kernelWidth].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Convolve2DBackwardWeights(this float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int? padding = null)
    {
        int batchSize = outputGradient.GetLength(0);
        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        Debug.Assert(kernelHeight == kernelWidth);
        int pad = padding ?? (kernelHeight / 2);

        float[,,,] paramGradient = new float[inputChannels, outputChannels, kernelHeight, kernelWidth];

        ref float inputRef = ref input[0, 0, 0, 0];
        ref float outputGradientRef = ref outputGradient[0, 0, 0, 0];

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref inputRef, input.Length);
        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradientRef, outputGradient.Length);
        Span<float> paramGradientSpan = MemoryMarshal.CreateSpan(ref paramGradient[0, 0, 0, 0], paramGradient.Length);

        for (int b = 0; b < batchSize; b++)
        {
            for (int ic = 0; ic < inputChannels; ic++)
            {
                for (int oc = 0; oc < outputChannels; oc++)
                {
                    for (int kh = 0; kh < kernelHeight; kh++)
                    {
                        for (int kw = 0; kw < kernelWidth; kw++)
                        {
                            float sum = 0f;
                            for (int oh = 0; oh < outputGradientHeight; oh++)
                            {
                                for (int ow = 0; ow < outputGradientWidth; ow++)
                                {
                                    int ih = oh + kh - pad;
                                    int iw = ow + kw - pad;
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                    {
                                        // sum += outputGradient[b, oc, oh, ow] * input[b, ic, ih, iw]
                                        sum += outputGradientSpan[b * (outputChannels * outputGradientHeight * outputGradientWidth) + oc * (outputGradientHeight * outputGradientWidth) + oh * outputGradientWidth + ow] *
                                               inputSpan[b * (inputChannels * inputHeight * inputWidth) + ic * (inputHeight * inputWidth) + ih * inputWidth + iw];
                                    }
                                }
                            }
                            // paramGradient[ic, oc, kh, kw] += sum;
                            paramGradientSpan[ic * (outputChannels * kernelHeight * kernelWidth) + oc * (kernelHeight * kernelWidth) + kh * kernelWidth + kw] += sum;
                        }
                    }
                }
            }
        }

        return paramGradient;
    }
}
