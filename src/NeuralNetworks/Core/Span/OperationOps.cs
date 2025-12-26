// Neural Networks in C♯
// File name: OperationOps.cs
// www.kowaliszyn.pl, 2025

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace NeuralNetworks.Core.Span;

public static class OperationOps
{
    /// <summary>
    /// 2D convolution forward pass on NHWC-like 4D tensors:
    /// Input: [batch, inChannels, inHeight, inWidth]
    /// Weights: [inChannels, outChannels, kernelHeight, kernelWidth]
    /// Output: [batch, outChannels, outHeight, outWidth]
    /// Padding is symmetric and computed as kernelSize / 2 (same padding) if not specified.
    /// </summary>
    /// <returns>Output</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Convolve2DForward(float[,,,] input, float[,,,] weights, int? padding = null)
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

        // pre-compute sizes for offsets
        int outputBSize = outputChannels * outputHeight * outputWidth;
        int outputCSize = outputHeight * outputWidth;
        int inputBSize = inputChannels * inputHeight * inputWidth;
        int inputCSize = inputHeight * inputWidth;
        int weightsCSize = outputChannels * kernelHeight * kernelWidth;
        int weightsOutputCSize = kernelHeight * kernelWidth;

        for (int b = 0; b < batchSize; b++)
        {
            int inputBIndex = b * inputBSize;
            int outputBIndex = b * outputBSize;
            for (int oc = 0; oc < outputChannels; oc++)
            {
                int weightsOutputCIndex = oc * weightsOutputCSize;
                int outputCIndex = oc * outputCSize;
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    int outputHIndex = oh * outputWidth;
                    int ohMinusPad = oh - pad;
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int owMinusPad = ow - pad;
                        float sum = 0f;
                        for (int ic = 0; ic < inputChannels; ic++)
                        {
                            int inputCIndex = ic * inputCSize;
                            int weightsInputCIndex = ic * weightsCSize;
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                int weightsKernelHIndex = kh * kernelWidth;
                                int ih = kh + ohMinusPad;
                                if (ih >= 0 && ih < inputHeight)
                                {
                                    int inputHIndex = ih * inputWidth;
                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {
                                        int iw = kw + owMinusPad;
                                        if (iw >= 0 && iw < inputWidth)
                                        {
                                            // sum += input[b, ic, ih, iw] * weights[ic, oc, kh, kw];
                                            sum += inputSpan[inputBIndex + inputCIndex + inputHIndex + iw] *
                                                   weightsSpan[weightsInputCIndex + weightsOutputCIndex + weightsKernelHIndex + kw];
                                        }
                                    }
                                }
                            }
                        }
                        // output[b, oc, oh, ow] = sum;
                        outputSpan[outputBIndex + outputCIndex + outputHIndex + ow] = sum;
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
    /// <returns>InputGradient</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Convolve2DBackwardInput(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int? padding = null)
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

        // pre-compute sizes for offsets
        int outputGradientSpanDim0 = outputChannels * outputGradientHeight * outputGradientWidth;
        int outputGradientSpanDim1 = outputGradientHeight * outputGradientWidth;
        int weightsSpanDim0 = outputChannels * kernelHeight * kernelWidth;
        int weightsSpanDim1 = kernelHeight * kernelWidth;
        int inputGradientSpanDim0 = inputChannels * inputHeight * inputWidth;
        int inputGradientSpanDim1 = inputHeight * inputWidth;

        for (int b = 0; b < batchSize; b++)
        {
            int bOffset = b * outputGradientSpanDim0;
            for (int ic = 0; ic < inputChannels; ic++)
            {
                int icOffset = ic * weightsSpanDim0;
                for (int ih = 0; ih < inputHeight; ih++)
                {
                    for (int iw = 0; iw < inputWidth; iw++)
                    {
                        float sum = 0f;
                        for (int oc = 0; oc < outputChannels; oc++)
                        {
                            int ocOffset = oc * outputGradientSpanDim1;
                            int ocWeightOffset = oc * weightsSpanDim1;
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                int oh = ih - kh + pad;
                                if (oh >= 0 && oh < outputGradientHeight)
                                {
                                    int ohOffset = oh * outputGradientWidth;
                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {

                                        int ow = iw - kw + pad;
                                        if (/*oh >= 0 && oh < outputGradientHeight &&*/ ow >= 0 && ow < outputGradientWidth)
                                        {
                                            // sum += outputGradient[b, oc, oh, ow] * weights[ic, oc, kh, kw];
                                            sum += outputGradientSpan[bOffset + ocOffset + ohOffset + ow] *
                                                   weightsSpan[icOffset + ocWeightOffset + kh * kernelWidth + kw];
                                        }
                                    }
                                }
                            }
                        }
                        // inputGradient[b, ic, ih, iw] = sum;
                        inputGradientSpan[b * inputGradientSpanDim0 + ic * inputGradientSpanDim1 + ih * inputWidth + iw] = sum;
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
    /// <returns>ParamGradient</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Convolve2DBackwardWeights(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int? padding = null)
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

        // pre-compute sizes for offsets

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

    /// <param name="outputGradient"></param>
    /// <param name="output"></param>
    /// <returns>InputGradient</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] MultiplyByTanhDerivative(float[,,,] outputGradient, float[,,,] output)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Tanh function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Tanh function tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) is 1 - tanh(x)^2.
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * (1 - tanh(x)^2).
        // The elementwise multiplication of the output gradient and the derivative of the Tanh function is returned as the input gradient.
        // tanh(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient

        int dim0 = outputGradient.GetLength(0);
        int dim1 = outputGradient.GetLength(1);
        int dim2 = outputGradient.GetLength(2);
        int dim3 = outputGradient.GetLength(3);

        Debug.Assert(dim0 > 0 && dim1 > 0 && dim2 > 0 && dim3 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(output.GetLength(0) == dim0 && output.GetLength(1) == dim1 && output.GetLength(2) == dim2 && output.GetLength(3) == dim3, "Shapes of outputGradient and output must match for elementwise operations.");

        float[,,,] result = new float[dim0, dim1, dim2, dim3];

        ref float ogRef = ref outputGradient[0, 0, 0, 0];
        ref float outRef = ref output[0, 0, 0, 0];
        ref float resRef = ref result[0, 0, 0, 0];

        ReadOnlySpan<float> ogSpan = MemoryMarshal.CreateReadOnlySpan(ref ogRef, outputGradient.Length);
        ReadOnlySpan<float> outSpan = MemoryMarshal.CreateReadOnlySpan(ref outRef, output.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref resRef, result.Length);

        for (int i = 0; i < resSpan.Length; i++)
        {
            float y = outSpan[i];
            resSpan[i] = ogSpan[i] * (1f - (y * y));
        }

        return result;
    }
}
