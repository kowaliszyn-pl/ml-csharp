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
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,,,] Convolve2DForward(float[,,,] input, float[,,,] weights, int? padding = null)
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

        Parallel.For(0, batchSize, b =>
        {
            int inputBIndex = b * inputBSize;
            int outputBIndex = b * outputBSize;
            Parallel.For(0, outputChannels, oc =>
            {
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
}


