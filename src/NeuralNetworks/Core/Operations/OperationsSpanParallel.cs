using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.Core.Operations;

internal class OperationsSpanParallel: OperationsSpan
{
    public override OperationBackendType BackendType => OperationBackendType.Cpu_Spans_Parallel;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,,,] Convolve2DCalcOutput(float[,,,] input, float[,,,] weights, int? padding = null)
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

        //ref float inputRef = ref input[0, 0, 0, 0];
        //ref float weightsRef = ref weights[0, 0, 0, 0];
        //ref float outputRef = ref output[0, 0, 0, 0];

        
        // pre-compute sizes for offsets
        int outputBSize = outputChannels * outputHeight * outputWidth;
        int outputCSize = outputHeight * outputWidth;
        int inputBSize = inputChannels * inputHeight * inputWidth;
        int inputCSize = inputHeight * inputWidth;
        int weightsCSize = outputChannels * kernelHeight * kernelWidth;
        int weightsOutputCSize = kernelHeight * kernelWidth;

        Parallel.For(0, batchSize, b =>
        {
            int inputBIndex = b * inputBSize;
            int outputBIndex = b * outputBSize;
            Parallel.For(0, outputChannels, oc =>
            {
                ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
                ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0, 0, 0], weights.Length);
                Span<float> outputSpan = MemoryMarshal.CreateSpan(ref output[0, 0, 0, 0], output.Length);

                int weightsOutputCIndex = oc * weightsOutputCSize;
                int outputCIndex = oc * outputCSize;
                for (int oh = 0; oh < outputHeight; oh++) // ~28
                {
                    int outputHIndex = oh * outputWidth;
                    int ohMinusPad = oh - pad;
                    for (int ow = 0; ow < outputWidth; ow++) // ~28
                    {
                        int owMinusPad = ow - pad;
                        float sum = 0f;
                        for (int ic = 0; ic < inputChannels; ic++) // 1 (black&white) or 3 (RGB)
                        {
                            int inputCIndex = ic * inputCSize;
                            int weightsInputCIndex = ic * weightsCSize;
                            for (int kh = 0; kh < kernelHeight; kh++) // ~3
                            {
                                int weightsKernelHIndex = kh * kernelWidth;
                                int ih = kh + ohMinusPad;
                                if (ih >= 0 && ih < inputHeight)
                                {
                                    int inputHIndex = ih * inputWidth;
                                    for (int kw = 0; kw < kernelWidth; kw++) // ~3
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
            });
        });

        return output;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,,,] Convolve2DBackwardInput(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int? padding = null)
    {
        int batchSize = outputGradient.GetLength(0);

        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputGradientChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        int kernelHeight = weights.GetLength(2);
        int kernelWidth = weights.GetLength(3);

        Debug.Assert(weights.GetLength(0) == inputChannels);
        Debug.Assert(weights.GetLength(1) == outputGradientChannels);
        Debug.Assert(kernelHeight == kernelWidth);

        int pad = padding ?? (kernelHeight / 2);

        float[,,,] inputGradient = new float[batchSize, inputChannels, inputHeight, inputWidth];

        //ref float weightsRef = ref weights[0, 0, 0, 0];
        //ref float outputGradientRef = ref outputGradient[0, 0, 0, 0];
        //ref float inputGradientRef = ref inputGradient[0, 0, 0, 0];

        

        // pre-compute sizes for offsets
        int outputGradientBSize = outputGradientChannels * outputGradientHeight * outputGradientWidth;
        int outputGradientCSize = outputGradientHeight * outputGradientWidth;
        int weightsInputCSize = outputGradientChannels * kernelHeight * kernelWidth;
        int weightsOutputCSize = kernelHeight * kernelWidth;
        int inputGradientBSize = inputChannels * inputHeight * inputWidth;
        int inputGradientCSize = inputHeight * inputWidth;

        Parallel.For(0, batchSize, b =>
        {
            int outputGradientBIndex = b * outputGradientBSize;
            Parallel.For(0, inputChannels, ic =>
            {
                ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weights[0, 0, 0, 0], weights.Length);
                ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
                Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradient[0, 0, 0, 0], inputGradient.Length);
                int weightsInputCIndex = ic * weightsInputCSize;
                int inputGradientCIndex = ic * inputGradientCSize;
                for (int ih = 0; ih < inputHeight; ih++)
                {
                    int inputGradientHIndex = ih * inputWidth;
                    int ihPlusPad = ih + pad;
                    for (int iw = 0; iw < inputWidth; iw++)
                    {
                        float sum = 0f;
                        int inputGradientBIndex = b * inputGradientBSize;
                        int iwPlusPad = iw + pad;
                        for (int oc = 0; oc < outputGradientChannels; oc++)
                        {
                            int outputGradientCIndex = oc * outputGradientCSize;
                            int weightsOutputCIndex = oc * weightsOutputCSize;
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                int oh = ihPlusPad - kh;
                                if (oh >= 0 && oh < outputGradientHeight)
                                {
                                    int weightsKernelHIndex = kh * kernelWidth;
                                    int outputHIndex = oh * outputGradientWidth;
                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {
                                        int ow = iwPlusPad - kw;
                                        if (ow >= 0 && ow < outputGradientWidth)
                                        {
                                            // sum += outputGradient[b, oc, oh, ow] * weights[ic, oc, kh, kw];
                                            sum += outputGradientSpan[outputGradientBIndex + outputGradientCIndex + outputHIndex + ow]
                                                * weightsSpan[weightsInputCIndex + weightsOutputCIndex + weightsKernelHIndex + kw];
                                        }
                                    }
                                }
                            }
                        }
                        // inputGradient[b, ic, ih, iw] = sum;
                        inputGradientSpan[inputGradientBIndex + inputGradientCIndex + inputGradientHIndex + iw] = sum;
                    }
                }
            });
        });

        return inputGradient;
    }

    /// <summary>
    /// Backward pass w.r.t. weights (parameters) for 2D convolution.
    /// Returns gradient with shape [inChannels, outChannels, kernelHeight, kernelWidth].
    /// </summary>
    /// <returns>ParamGradient</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,,,] Convolve2DBackwardWeights(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int? padding = null)
    {
        int batchSize = outputGradient.GetLength(0);

        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputGradientChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        Debug.Assert(kernelHeight == kernelWidth);
        int pad = padding ?? (kernelHeight / 2);

        float[,,,] paramGradient = new float[inputChannels, outputGradientChannels, kernelHeight, kernelWidth];

        //ref float inputRef = ref input[0, 0, 0, 0];
        //ref float outputGradientRef = ref outputGradient[0, 0, 0, 0];

        

        // pre-compute sizes for offsets
        int outputGradientBSize = outputGradientChannels * outputGradientHeight * outputGradientWidth;
        int outputGradientOutputCSize = outputGradientHeight * outputGradientWidth;
        int inputBSize = inputChannels * inputHeight * inputWidth;
        int inputCSize = inputHeight * inputWidth;
        int paramGradientInputCSize = outputGradientChannels * kernelHeight * kernelWidth;
        int paramGradientOutputCSize = kernelHeight * kernelWidth;

        Parallel.For(0, batchSize, b =>
        {
            int outputGradientBIndex = b * outputGradientBSize;
            int inputBIndex = b * inputBSize;
            Parallel.For(0, inputChannels, ic =>
            {
                ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref input[0, 0, 0, 0], input.Length);
                ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0, 0, 0], outputGradient.Length);
                Span<float> paramGradientSpan = MemoryMarshal.CreateSpan(ref paramGradient[0, 0, 0, 0], paramGradient.Length);
                int inputCIndex = ic * inputCSize;
                int paramGradientInputCIndex = ic * paramGradientInputCSize;

                for (int oc = 0; oc < outputGradientChannels; oc++)
                {
                    int outputGradientOutputCIndex = oc * outputGradientOutputCSize;
                    int paramGradientOutputCIndex = oc * paramGradientOutputCSize;
                    for (int kh = 0; kh < kernelHeight; kh++)
                    {
                        int paramGradientKernelHIndex = kh * kernelWidth;
                        int khMinusPad = kh - pad;
                        for (int kw = 0; kw < kernelWidth; kw++)
                        {
                            int kwMinusPad = kw - pad;
                            float sum = 0f;
                            for (int oh = 0; oh < outputGradientHeight; oh++)
                            {
                                int ih = oh + khMinusPad;
                                if (ih >= 0 && ih < inputHeight)
                                {
                                    int inputHIndex = ih * inputWidth;
                                    int outputGradientHIndex = oh * outputGradientWidth;
                                    for (int ow = 0; ow < outputGradientWidth; ow++)
                                    {
                                        int iw = ow + kwMinusPad;
                                        if (iw >= 0 && iw < inputWidth)
                                        {
                                            // sum += outputGradient[b, oc, oh, ow] * input[b, ic, ih, iw]
                                            sum += outputGradientSpan[outputGradientBIndex + outputGradientOutputCIndex + outputGradientHIndex + ow] *
                                                   inputSpan[inputBIndex + inputCIndex + inputHIndex + iw];
                                        }
                                    }
                                }
                            }
                            // paramGradient[ic, oc, kh, kw] += sum;
                            paramGradientSpan[paramGradientInputCIndex + paramGradientOutputCIndex + paramGradientKernelHIndex + kw] += sum;
                        }
                    }
                }
            });
        });

        return paramGradient;
    }


}


