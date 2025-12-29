// Neural Networks in C♯
// File name: OperationsGpu.cs
// www.kowaliszyn.pl, 2025

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

namespace NeuralNetworks.Core.Operations;

internal class OperationsGpu : OperationsSpanParallel, IDisposable
{
    private readonly Context s_context;
    private readonly Accelerator s_accelerator;
    private bool _disposedValue;

    public OperationsGpu()
    {
        // Initialize ILGPU context and accelerator
        s_context = Context.Create(builder => builder.Cuda().CPU());
        s_accelerator = s_context.GetPreferredDevice(preferCPU: false).CreateAccelerator(s_context);
    }

    public override OperationBackendType BackendType => OperationBackendType.Gpu;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,] WeightMultiplyCalcOutput(float[,] input, float[,] weights)
    {
        int batchSize = input.GetLength(0);
        int inputFeatures = input.GetLength(1);
        int outputFeatures = weights.GetLength(1);

        Debug.Assert(weights.GetLength(0) == inputFeatures, "Input features must match weight input dimension.");

        float[,] output = new float[batchSize, outputFeatures];

        using MemoryBuffer2D<float, Stride2D.DenseX> inputDev = s_accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, inputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> weightsDev = s_accelerator.Allocate2DDenseX<float>(new Index2D(inputFeatures, outputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> outputDev = s_accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, outputFeatures));

        inputDev.View.CopyFromCPU(input);
        weightsDev.View.CopyFromCPU(weights);

        Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int> kernel =
            s_accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>(WeightMultiplyOutputKernel);

        kernel(new Index2D(batchSize, outputFeatures), inputDev.View, weightsDev.View, outputDev.View, inputFeatures);
        s_accelerator.Synchronize();

        outputDev.View.CopyToCPU(output);
        return output;
    }

    private static void WeightMultiplyOutputKernel(Index2D index, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> output, int kDim)
    {
        int row = index.X;
        int col = index.Y;

        if (row >= output.Extent.X || col >= output.Extent.Y)
        {
            return;
        }

        float sum = 0f;
        for (int k = 0; k < kDim; k++)
        {
            sum += input[row, k] * weights[k, col];
        }

        output[row, col] = sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,] WeightMultiplyCalcInputGradient(float[,] outputGradient, float[,] weights)
    {
        int batchSize = outputGradient.GetLength(0);
        int outputFeatures = outputGradient.GetLength(1);
        int inputFeatures = weights.GetLength(0);

        Debug.Assert(outputFeatures == weights.GetLength(1), "Output features of output gradient must match weight output dimension.");

        float[,] inputGradient = new float[batchSize, inputFeatures];

        using MemoryBuffer2D<float, Stride2D.DenseX> outputGradientDev = s_accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, outputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> weightsDev = s_accelerator.Allocate2DDenseX<float>(new Index2D(inputFeatures, outputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> inputGradientDev = s_accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, inputFeatures));

        outputGradientDev.View.CopyFromCPU(outputGradient);
        weightsDev.View.CopyFromCPU(weights);

        Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int> kernel =
            s_accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>(WeightMultiplyInputGradientKernel);

        kernel(new Index2D(batchSize, inputFeatures), outputGradientDev.View, weightsDev.View, inputGradientDev.View, outputFeatures);
        s_accelerator.Synchronize();

        inputGradientDev.View.CopyToCPU(inputGradient);
        return inputGradient;
    }

    private static void WeightMultiplyInputGradientKernel(Index2D index, ArrayView2D<float, Stride2D.DenseX> outputGradient, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> inputGradient, int kDim)
    {
        int row = index.X;
        int col = index.Y;

        if (row >= inputGradient.Extent.X || col >= inputGradient.Extent.Y)
        {
            return;
        }

        float sum = 0f;
        for (int k = 0; k < kDim; k++)
        {
            sum += outputGradient[row, k] * weights[col, k];
        }

        inputGradient[row, col] = sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,] WeightMultiplyCalcParamGradient(float[,] input, float[,] outputGradient)
    {
        int batchSize = input.GetLength(0);
        int inputFeatures = input.GetLength(1);
        int outputFeatures = outputGradient.GetLength(1);

        Debug.Assert(outputGradient.GetLength(0) == batchSize, "Batch size of output gradient must match batch size of input.");

        float[,] paramGradient = new float[inputFeatures, outputFeatures];

        using MemoryBuffer2D<float, Stride2D.DenseX> inputDev = s_accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, inputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> outputGradientDev = s_accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, outputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> paramGradientDev = s_accelerator.Allocate2DDenseX<float>(new Index2D(inputFeatures, outputFeatures));

        inputDev.View.CopyFromCPU(input);
        outputGradientDev.View.CopyFromCPU(outputGradient);

        Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> kernel =
            s_accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(WeightMultiplyParamGradientKernel);

        kernel(new Index2D(inputFeatures, outputFeatures), inputDev.View, outputGradientDev.View, paramGradientDev.View);
        s_accelerator.Synchronize();

        paramGradientDev.View.CopyToCPU(paramGradient);
        return paramGradient;
    }

    private static void WeightMultiplyParamGradientKernel(Index2D index, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> outputGradient, ArrayView2D<float, Stride2D.DenseX> paramGradient)
    {
        int row = index.X;
        int col = index.Y;

        if (row >= paramGradient.Extent.X || col >= paramGradient.Extent.Y)
        {
            return;
        }

        long batch = input.Extent.X;
        float sum = 0f;
        for (int b = 0; b < batch; b++)
        {
            sum += input[b, row] * outputGradient[b, col];
        }

        paramGradient[row, col] = sum;
    }

    // ... existing usings and class header stay as is ...

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

        int outputHeight = inputHeight - kernelHeight + 1 + (2 * pad);
        int outputWidth = inputWidth - kernelWidth + 1 + (2 * pad);

        float[,,,] output = new float[batchSize, outputChannels, outputHeight, outputWidth];

        // Layout:
        // inputDev:  (batch * inChannels, inHeight, inWidth)
        // weightsDev:(inChannels * outChannels, kH, kW)
        // outputDev: (batch * outChannels, outHeight, outWidth)
        using MemoryBuffer3D<float, Stride3D.DenseXY> inputDev =
            s_accelerator.Allocate3DDenseXY<float>(new Index3D(batchSize * inputChannels, inputHeight, inputWidth));
        using MemoryBuffer3D<float, Stride3D.DenseXY> weightsDev =
            s_accelerator.Allocate3DDenseXY<float>(new Index3D(inputChannels * outputChannels, kernelHeight, kernelWidth));
        using MemoryBuffer3D<float, Stride3D.DenseXY> outputDev =
            s_accelerator.Allocate3DDenseXY<float>(new Index3D(batchSize * outputChannels, outputHeight, outputWidth));

        ref float inputRef = ref input[0, 0, 0, 0];
        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref inputRef, input.Length);
        //inputDev.View.CopyFromCPU(ref inputRef, input.Length);

        CopyInput4DTo3D(input, inputDev.View);
        CopyWeights4DTo3D(weights, weightsDev.View);

        Action<
            Index3D,
            ArrayView3D<float, Stride3D.DenseXY>,
            ArrayView3D<float, Stride3D.DenseXY>,
            ArrayView3D<float, Stride3D.DenseXY>,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            int> fwdKernel =
            s_accelerator.LoadAutoGroupedStreamKernel<
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int>(Convolve2DForwardKernel);

        // We encode: X = batch * outChannels + oc, Y = oh, Z = ow
        fwdKernel(
            new Index3D(batchSize * outputChannels, outputHeight, outputWidth),
            inputDev.View,
            weightsDev.View,
            outputDev.View,
            pad,
            inputChannels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
            outputChannels,
            batchSize);
        s_accelerator.Synchronize();

        CopyOutput3DTo4D(outputDev.View, output);
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

        using MemoryBuffer3D<float, Stride3D.DenseXY> weightsDev =
            s_accelerator.Allocate3DDenseXY<float>(new Index3D(inputChannels * outputGradientChannels, kernelHeight, kernelWidth));
        using MemoryBuffer3D<float, Stride3D.DenseXY> outputGradientDev =
            s_accelerator.Allocate3DDenseXY<float>(new Index3D(batchSize * outputGradientChannels, outputGradientHeight, outputGradientWidth));
        using MemoryBuffer3D<float, Stride3D.DenseXY> inputGradientDev =
            s_accelerator.Allocate3DDenseXY<float>(new Index3D(batchSize * inputChannels, inputHeight, inputWidth));

        CopyWeights4DTo3D(weights, weightsDev.View);
        CopyOutputGrad4DTo3D(outputGradient, outputGradientDev.View);

        Action<
            Index3D,
            ArrayView3D<float, Stride3D.DenseXY>,
            ArrayView3D<float, Stride3D.DenseXY>,
            ArrayView3D<float, Stride3D.DenseXY>,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            int> bwdInputKernel =
            s_accelerator.LoadAutoGroupedStreamKernel<
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int>(Convolve2DBackwardInputKernel);

        // Encode: X = batch * inChannels + ic, Y = ih, Z = iw
        bwdInputKernel(
            new Index3D(batchSize * inputChannels, inputHeight, inputWidth),
            outputGradientDev.View,
            weightsDev.View,
            inputGradientDev.View,
            pad,
            inputChannels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
            outputGradientChannels,
            batchSize);
        s_accelerator.Synchronize();

        CopyInputGrad3DTo4D(inputGradientDev.View, inputGradient);
        return inputGradient;
    }

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

        using MemoryBuffer3D<float, Stride3D.DenseXY> inputDev =
            s_accelerator.Allocate3DDenseXY<float>(new Index3D(batchSize * inputChannels, inputHeight, inputWidth));
        using MemoryBuffer3D<float, Stride3D.DenseXY> outputGradientDev =
            s_accelerator.Allocate3DDenseXY<float>(new Index3D(batchSize * outputGradientChannels, outputGradientHeight, outputGradientWidth));
        using MemoryBuffer3D<float, Stride3D.DenseXY> paramGradientDev =
            s_accelerator.Allocate3DDenseXY<float>(new Index3D(inputChannels * outputGradientChannels, kernelHeight, kernelWidth));

        CopyInput4DTo3D(input, inputDev.View);
        CopyOutputGrad4DTo3D(outputGradient, outputGradientDev.View);
        // paramGradientDev.View.MemSetToZero();

        Action<
            Index3D,
            ArrayView3D<float, Stride3D.DenseXY>,
            ArrayView3D<float, Stride3D.DenseXY>,
            ArrayView3D<float, Stride3D.DenseXY>,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            int> bwdWeightsKernel =
            s_accelerator.LoadAutoGroupedStreamKernel<
                Index3D,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>,
                ArrayView3D<float, Stride3D.DenseXY>,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int>(Convolve2DBackwardWeightsKernel);

        // Encode: X = inChannels * outChannels + (ic * outChannels + oc) flattens to (ic, oc, kh, kw) via division/mod
        // But simpler is: X = (ic * outChannels) + oc, Y = kh, Z = kw
        bwdWeightsKernel(
            new Index3D(inputChannels * outputGradientChannels, kernelHeight, kernelWidth),
            inputDev.View,
            outputGradientDev.View,
            paramGradientDev.View,
            pad,
            inputChannels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
            outputGradientChannels,
            batchSize);
        s_accelerator.Synchronize();

        CopyWeights3DTo4D(paramGradientDev.View, paramGradient);
        return paramGradient;
    }

    #region GPU Kernels - Convolution

    // Forward: each thread computes one output element [b, oc, oh, ow]
    // Encoded as: flatBoc = index.X = b * outputChannels + oc
    // oc = flatBoc % outputChannels; b = flatBoc / outputChannels
    // oh = index.Y; ow = index.Z
    private static void Convolve2DForwardKernel(
        Index3D index,
        ArrayView3D<float, Stride3D.DenseXY> input,
        ArrayView3D<float, Stride3D.DenseXY> weights,
        ArrayView3D<float, Stride3D.DenseXY> output,
        int pad,
        int inputChannels,
        int inputHeight,
        int inputWidth,
        int kernelHeight,
        int kernelWidth,
        int outputChannels,
        int batchSize)
    {
        int flatBoc = index.X;
        int oh = index.Y;
        int ow = index.Z;

        int outputHeight = (int)output.Extent.Y;
        int outputWidth = (int)output.Extent.Z;

        if (oh >= outputHeight || ow >= outputWidth)
        {
            return;
        }

        int b = flatBoc / outputChannels;
        int oc = flatBoc - (b * outputChannels);

        if (b >= batchSize)
        {
            return;
        }

        int ohMinusPad = oh - pad;
        int owMinusPad = ow - pad;

        float sum = 0f;

        for (int ic = 0; ic < inputChannels; ic++)
        {
            int bc = (b * inputChannels) + ic;
            int ioc = (ic * outputChannels) + oc;

            for (int kh = 0; kh < kernelHeight; kh++)
            {
                int ih = kh + ohMinusPad;
                if (ih < 0 || ih >= inputHeight)
                {
                    continue;
                }

                for (int kw = 0; kw < kernelWidth; kw++)
                {
                    int iw = kw + owMinusPad;
                    if (iw < 0 || iw >= inputWidth)
                    {
                        continue;
                    }

                    float inputVal = input[bc, ih, iw];
                    float weightVal = weights[ioc, kh, kw];
                    sum += inputVal * weightVal;
                }
            }
        }

        output[flatBoc, oh, ow] = sum;
    }

    // Backward input: each thread computes one input gradient element [b, ic, ih, iw]
    // Encoded as: flatBic = index.X = b * inputChannels + ic
    // ic = flatBic % inputChannels; b = flatBic / inputChannels
    // ih = index.Y; iw = index.Z
    private static void Convolve2DBackwardInputKernel(
        Index3D index,
        ArrayView3D<float, Stride3D.DenseXY> outputGradient,
        ArrayView3D<float, Stride3D.DenseXY> weights,
        ArrayView3D<float, Stride3D.DenseXY> inputGradient,
        int pad,
        int inputChannels,
        int inputHeight,
        int inputWidth,
        int kernelHeight,
        int kernelWidth,
        int outputChannels,
        int batchSize)
    {
        int flatBic = index.X;
        int ih = index.Y;
        int iw = index.Z;

        if (ih >= inputHeight || iw >= inputWidth)
        {
            return;
        }

        int b = flatBic / inputChannels;
        int ic = flatBic - (b * inputChannels);

        if (b >= batchSize)
        {
            return;
        }

        int ihPlusPad = ih + pad;
        int iwPlusPad = iw + pad;

        float sum = 0f;

        int outputGradientHeight = (int)outputGradient.Extent.Y;
        int outputGradientWidth = (int)outputGradient.Extent.Z;

        for (int oc = 0; oc < outputChannels; oc++)
        {
            int boc = (b * outputChannels) + oc;
            int ioc = (ic * outputChannels) + oc;

            for (int kh = 0; kh < kernelHeight; kh++)
            {
                int oh = ihPlusPad - kh;
                if (oh < 0 || oh >= outputGradientHeight)
                {
                    continue;
                }

                for (int kw = 0; kw < kernelWidth; kw++)
                {
                    int ow = iwPlusPad - kw;
                    if (ow < 0 || ow >= outputGradientWidth)
                    {
                        continue;
                    }

                    float ogVal = outputGradient[boc, oh, ow];
                    float weightVal = weights[ioc, kh, kw];
                    sum += ogVal * weightVal;
                }
            }
        }

        inputGradient[flatBic, ih, iw] = sum;
    }

    // Backward weights: each thread computes one weight gradient element [ic, oc, kh, kw]
    // Encoded as: flatIoc = index.X = ic * outputChannels + oc
    // ic = flatIoc / outputChannels; oc = flatIoc % outputChannels
    // kh = index.Y; kw = index.Z
    private static void Convolve2DBackwardWeightsKernel(
        Index3D index,
        ArrayView3D<float, Stride3D.DenseXY> input,
        ArrayView3D<float, Stride3D.DenseXY> outputGradient,
        ArrayView3D<float, Stride3D.DenseXY> paramGradient,
        int pad,
        int inputChannels,
        int inputHeight,
        int inputWidth,
        int kernelHeight,
        int kernelWidth,
        int outputChannels,
        int batchSize)
    {
        int flatIoc = index.X;
        int kh = index.Y;
        int kw = index.Z;

        if (kh >= kernelHeight || kw >= kernelWidth)
        {
            return;
        }

        int ic = flatIoc / outputChannels;
        int oc = flatIoc - (ic * outputChannels);

        if (ic >= inputChannels)
        {
            return;
        }

        int khMinusPad = kh - pad;
        int kwMinusPad = kw - pad;

        float sum = 0f;

        int outputGradientHeight = (int)outputGradient.Extent.Y;
        int outputGradientWidth = (int)outputGradient.Extent.Z;

        for (int b = 0; b < batchSize; b++)
        {
            int bc = (b * inputChannels) + ic;
            int boc = (b * outputChannels) + oc;

            for (int oh = 0; oh < outputGradientHeight; oh++)
            {
                int ih = oh + khMinusPad;
                if (ih < 0 || ih >= inputHeight)
                {
                    continue;
                }

                for (int ow = 0; ow < outputGradientWidth; ow++)
                {
                    int iw = ow + kwMinusPad;
                    if (iw < 0 || iw >= inputWidth)
                    {
                        continue;
                    }

                    float ogVal = outputGradient[boc, oh, ow];
                    float inVal = input[bc, ih, iw];
                    sum += ogVal * inVal;
                }
            }
        }

        paramGradient[flatIoc, kh, kw] = sum;
    }

    #endregion

    #region Host-side 4D<->3D helpers

    private static void CopyInput4DTo3D(float[,,,] src, ArrayView3D<float, Stride3D.DenseXY> dst)
    {
        int batchSize = src.GetLength(0);
        int channels = src.GetLength(1);
        int height = src.GetLength(2);
        int width = src.GetLength(3);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int bc = (b * channels) + c;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        dst[bc, h, w] = src[b, c, h, w];
                    }
                }
            }
        }
    }

    private static void CopyOutput3DTo4D(ArrayView3D<float, Stride3D.DenseXY> src, float[,,,] dst)
    {
        int batchSize = dst.GetLength(0);
        int channels = dst.GetLength(1);
        int height = dst.GetLength(2);
        int width = dst.GetLength(3);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int bc = (b * channels) + c;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        dst[b, c, h, w] = src[bc, h, w];
                    }
                }
            }
        }
    }

    private static void CopyWeights4DTo3D(float[,,,] src, ArrayView3D<float, Stride3D.DenseXY> dst)
    {
        int inChannels = src.GetLength(0);
        int outChannels = src.GetLength(1);
        int kH = src.GetLength(2);
        int kW = src.GetLength(3);

        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                int ioc = (ic * outChannels) + oc;
                for (int kh = 0; kh < kH; kh++)
                {
                    for (int kw = 0; kw < kW; kw++)
                    {
                        dst[ioc, kh, kw] = src[ic, oc, kh, kw];
                    }
                }
            }
        }
    }

    private static void CopyWeights3DTo4D(ArrayView3D<float, Stride3D.DenseXY> src, float[,,,] dst)
    {
        int inChannels = dst.GetLength(0);
        int outChannels = dst.GetLength(1);
        int kH = dst.GetLength(2);
        int kW = dst.GetLength(3);

        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                int ioc = (ic * outChannels) + oc;
                for (int kh = 0; kh < kH; kh++)
                {
                    for (int kw = 0; kw < kW; kw++)
                    {
                        dst[ic, oc, kh, kw] = src[ioc, kh, kw];
                    }
                }
            }
        }
    }

    private static void CopyOutputGrad4DTo3D(float[,,,] src, ArrayView3D<float, Stride3D.DenseXY> dst)
    {
        int batchSize = src.GetLength(0);
        int channels = src.GetLength(1);
        int height = src.GetLength(2);
        int width = src.GetLength(3);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int bc = (b * channels) + c;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        dst[bc, h, w] = src[b, c, h, w];
                    }
                }
            }
        }
    }

    private static void CopyInputGrad3DTo4D(ArrayView3D<float, Stride3D.DenseXY> src, float[,,,] dst)
    {
        int batchSize = dst.GetLength(0);
        int channels = dst.GetLength(1);
        int height = dst.GetLength(2);
        int width = dst.GetLength(3);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int bc = (b * channels) + c;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        dst[b, c, h, w] = src[bc, h, w];
                    }
                }
            }
        }
    }

    #endregion

    #region Dispose pattern

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            if (disposing)
            {
                // TODO: dispose managed state (managed objects)
            }

            // Free unmanaged resources (unmanaged objects) and override finalizer
            s_accelerator.Dispose();
            s_context.Dispose();
            // TODO: set large fields to null
            _disposedValue = true;
        }
    }

    // Override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
    ~OperationsGpu()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: false);
    }

    void IDisposable.Dispose()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    #endregion
}
