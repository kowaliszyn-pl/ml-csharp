// Neural Networks in C♯
// File name: OperationsGpu.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

namespace NeuralNetworks.Core.Operations;

using FloatDense1DView = ArrayView1D<float, Stride1D.Dense>;
using FloatDense2DView = ArrayView2D<float, Stride2D.DenseX>;

public readonly struct Convolve2DOutputMeta
{
    public Convolve2DOutputMeta(
        int pad,
        int inputChannels,
        int inputHeight,
        int inputWidth,
        int kernelHeight,
        int kernelWidth,
        int outputChannels,
        int batchSize,
        int outputHeight,
        int outputWidth,
        int inputBatchSize,
        int inputChannelSize,
        int weightsChannelSize,
        int weightsOutputChannelSize,
        int outputBatchSize,
        int outputChannelSize)
    {
        Pad = pad;
        InputChannels = inputChannels;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        KernelHeight = kernelHeight;
        KernelWidth = kernelWidth;
        OutputChannels = outputChannels;
        BatchSize = batchSize;
        OutputHeight = outputHeight;
        OutputWidth = outputWidth;
        InputBatchSize = inputBatchSize;
        InputChannelSize = inputChannelSize;
        WeightsChannelSize = weightsChannelSize;
        WeightsOutputChannelSize = weightsOutputChannelSize;
        OutputBatchSize = outputBatchSize;
        OutputChannelSize = outputChannelSize;
    }

    public int Pad { get; }
    public int InputChannels { get; }
    public int InputHeight { get; }
    public int InputWidth { get; }
    public int KernelHeight { get; }
    public int KernelWidth { get; }
    public int OutputChannels { get; }
    public int BatchSize { get; }
    public int OutputHeight { get; }
    public int OutputWidth { get; }
    public int InputBatchSize { get; }
    public int InputChannelSize { get; }
    public int WeightsChannelSize { get; }
    public int WeightsOutputChannelSize { get; }
    public int OutputBatchSize { get; }
    public int OutputChannelSize { get; }
}

internal class OperationsGpu : OperationsSpanParallel, IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;

    private readonly Action<Index3D, FloatDense1DView, FloatDense1DView, FloatDense1DView, Convolve2DOutputMeta> _convolve2DOutputKernel;
    private readonly Action<Index2D, FloatDense2DView, FloatDense2DView, FloatDense2DView, int> _weightMultiplyCalcOutputKernel;
    private readonly Action<Index2D, FloatDense2DView, FloatDense2DView, FloatDense2DView, int> _weightMultiplyInputGradientKernel;
    private readonly Action<Index2D, FloatDense2DView, FloatDense2DView, FloatDense2DView> _weightMultiplyParamGradientKernel;

    private bool _disposedValue;

    public OperationsGpu()
    {
        // Initialize ILGPU context and accelerator
        _context = Context.Create(builder => builder.Cuda().CPU());
        _accelerator = _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        _convolve2DOutputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, FloatDense1DView, FloatDense1DView, FloatDense1DView, Convolve2DOutputMeta>(Convolve2DOutputKernel);
        _weightMultiplyCalcOutputKernel =
           _accelerator.LoadAutoGroupedStreamKernel<Index2D, FloatDense2DView, FloatDense2DView, FloatDense2DView, int>(WeightMultiplyCalcOutputKernel);
        _weightMultiplyInputGradientKernel =
            _accelerator.LoadAutoGroupedStreamKernel<Index2D, FloatDense2DView, FloatDense2DView, FloatDense2DView, int>(WeightMultiplyInputGradientKernel);
        _weightMultiplyParamGradientKernel =
            _accelerator.LoadAutoGroupedStreamKernel<Index2D, FloatDense2DView, FloatDense2DView, FloatDense2DView>(WeightMultiplyParamGradientKernel);
    }

    public override OperationBackendType BackendType => OperationBackendType.Gpu;

    MemoryBuffer2D<float, Stride2D.DenseX> inputDev;
    MemoryBuffer2D<float, Stride2D.DenseX> weightsDev;
    MemoryBuffer2D<float, Stride2D.DenseX> outputDev;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,] WeightMultiplyOutput(float[,] input, float[,] weights)
    {
        int batchSize = input.GetLength(0);

        int inputFeatures = input.GetLength(1);
        int outputFeatures = weights.GetLength(1);

        Debug.Assert(weights.GetLength(0) == inputFeatures, "Input features must match weight input dimension.");

        float[,] output = new float[batchSize, outputFeatures];

        using MemoryBuffer2D<float, Stride2D.DenseX> inputDev = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, inputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> weightsDev = _accelerator.Allocate2DDenseX<float>(new Index2D(inputFeatures, outputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> outputDev = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, outputFeatures));

        inputDev.View.CopyFromCPU(input);
        weightsDev.View.CopyFromCPU(weights);

        _weightMultiplyCalcOutputKernel(new Index2D(batchSize, outputFeatures), inputDev.View, weightsDev.View, outputDev.View, inputFeatures);
        _accelerator.Synchronize();

        outputDev.View.CopyToCPU(output);
        return output;
    }

    private static void WeightMultiplyCalcOutputKernel(Index2D index, FloatDense2DView input, FloatDense2DView weights, FloatDense2DView output, int kDim)
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
    public override float[,] WeightMultiplyInputGradient(float[,] outputGradient, float[,] weights)
    {
        int batchSize = outputGradient.GetLength(0);
        int outputFeatures = outputGradient.GetLength(1);
        int inputFeatures = weights.GetLength(0);

        Debug.Assert(outputFeatures == weights.GetLength(1), "Output features of output gradient must match weight output dimension.");

        float[,] inputGradient = new float[batchSize, inputFeatures];

        using MemoryBuffer2D<float, Stride2D.DenseX> outputGradientDev = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, outputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> weightsDev = _accelerator.Allocate2DDenseX<float>(new Index2D(inputFeatures, outputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> inputGradientDev = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, inputFeatures));

        outputGradientDev.View.CopyFromCPU(outputGradient);
        weightsDev.View.CopyFromCPU(weights);

        //Action<Index2D, FloatDense2DView, FloatDense2DView, FloatDense2DView, int> kernel =
        //    _accelerator.LoadAutoGroupedStreamKernel<Index2D, FloatDense2DView, FloatDense2DView, FloatDense2DView, int>(WeightMultiplyInputGradientKernel);

        _weightMultiplyInputGradientKernel(new Index2D(batchSize, inputFeatures), outputGradientDev.View, weightsDev.View, inputGradientDev.View, outputFeatures);
        _accelerator.Synchronize();

        inputGradientDev.View.CopyToCPU(inputGradient);
        return inputGradient;
    }

    private static void WeightMultiplyInputGradientKernel(Index2D index, FloatDense2DView outputGradient, FloatDense2DView weights, FloatDense2DView inputGradient, int kDim)
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
    public override float[,] WeightMultiplyParamGradient(float[,] input, float[,] outputGradient)
    {
        int batchSize = input.GetLength(0);
        int inputFeatures = input.GetLength(1);
        int outputFeatures = outputGradient.GetLength(1);

        Debug.Assert(outputGradient.GetLength(0) == batchSize, "Batch size of output gradient must match batch size of input.");

        float[,] paramGradient = new float[inputFeatures, outputFeatures];

        using MemoryBuffer2D<float, Stride2D.DenseX> inputDev = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, inputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> outputGradientDev = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, outputFeatures));
        using MemoryBuffer2D<float, Stride2D.DenseX> paramGradientDev = _accelerator.Allocate2DDenseX<float>(new Index2D(inputFeatures, outputFeatures));

        inputDev.View.CopyFromCPU(input);
        outputGradientDev.View.CopyFromCPU(outputGradient);

        //Action<Index2D, FloatDense2DView, FloatDense2DView, FloatDense2DView> kernel =
         //   _accelerator.LoadAutoGroupedStreamKernel<Index2D, FloatDense2DView, FloatDense2DView, FloatDense2DView>(WeightMultiplyParamGradientKernel);

        _weightMultiplyParamGradientKernel(new Index2D(inputFeatures, outputFeatures), inputDev.View, outputGradientDev.View, paramGradientDev.View);
        _accelerator.Synchronize();

        paramGradientDev.View.CopyToCPU(paramGradient);
        return paramGradient;
    }

    private static void WeightMultiplyParamGradientKernel(Index2D index, FloatDense2DView input, FloatDense2DView outputGradient, FloatDense2DView paramGradient)
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
    /*
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,,,] Convolve2DOutput(float[,,,] input, float[,,,] weights, int? padding = null)
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

        int inputChannelSize = inputHeight * inputWidth;
        int inputBatchSize = inputChannels * inputChannelSize;
        int weightsOutputChannelSize = kernelHeight * kernelWidth;
        int weightsChannelSize = outputChannels * weightsOutputChannelSize;
        int outputChannelSize = outputHeight * outputWidth;
        int outputBatchSize = outputChannels * outputChannelSize;

        using MemoryBuffer1D<float, Stride1D.Dense> inputDev =
            _accelerator.Allocate1D<float>(new Index1D(batchSize * inputBatchSize));
        using MemoryBuffer1D<float, Stride1D.Dense> weightsDev =
            _accelerator.Allocate1D<float>(new Index1D(weightChannels * weightsChannelSize));
        using MemoryBuffer1D<float, Stride1D.Dense> outputDev =
            _accelerator.Allocate1D<float>(new Index1D(batchSize * outputBatchSize));

        inputDev.View.CopyFromCPU(ref input[0, 0, 0, 0], input.Length);
        weightsDev.View.CopyFromCPU(ref weights[0, 0, 0, 0], weights.Length);

        Convolve2DOutputMeta meta = new(
            pad,
            inputChannels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
            outputChannels,
            batchSize,
            outputHeight,
            outputWidth,
            inputBatchSize,
            inputChannelSize,
            weightsChannelSize,
            weightsOutputChannelSize,
            outputBatchSize,
            outputChannelSize);

        // Action<Index3D, FloatDense1DView, FloatDense1DView, FloatDense1DView, Convolve2DOutputMeta> _convolve2DOutputKernel =
        _accelerator.LoadAutoGroupedStreamKernel<Index3D, FloatDense1DView, FloatDense1DView, FloatDense1DView, Convolve2DOutputMeta>(Convolve2DOutputKernel);

        _convolve2DOutputKernel(
                new Index3D(batchSize * outputChannels, outputHeight, outputWidth),
                inputDev.View,
                weightsDev.View,
                outputDev.View,
                meta);

        _accelerator.Synchronize();
        outputDev.View.CopyToCPU(ref output[0, 0, 0, 0], output.Length);
        return output;
    }
    */
    // Forward: each thread computes one output element [batchSize * outputChannels * outputHeight * outputWidth]
    // Encoded as: batchOutputChannelIndex = index.X = b * outputChannels + oc
    // oc = batchOutputChannelIndex % outputChannels; b = batchOutputChannelIndex / outputChannels
    // oh = index.Y; ow = index.Z
    private static void Convolve2DOutputKernel(Index3D index, FloatDense1DView input, FloatDense1DView weights, FloatDense1DView output, Convolve2DOutputMeta meta)
    {
        int pad = meta.Pad;
        int inputChannels = meta.InputChannels;
        int inputHeight = meta.InputHeight;
        int inputWidth = meta.InputWidth;
        int kernelHeight = meta.KernelHeight;
        int kernelWidth = meta.KernelWidth;
        int outputChannels = meta.OutputChannels;
        int batchSize = meta.BatchSize;
        int outputHeight = meta.OutputHeight;
        int outputWidth = meta.OutputWidth;

        int batchOutputChannelIndex = index.X;
        int oh = index.Y;
        int ow = index.Z;

        if (oh >= outputHeight || ow >= outputWidth)
        {
            return;
        }

        int b = batchOutputChannelIndex / outputChannels;
        if (b >= batchSize)
        {
            return;
        }

        int oc = batchOutputChannelIndex - (b * outputChannels);

        int outputBIndex = b * meta.OutputBatchSize;
        int outputCIndex = oc * meta.OutputChannelSize;
        int outputHIndex = oh * outputWidth;

        int inputBIndex = b * meta.InputBatchSize;
        int weightsOutputCIndex = oc * meta.WeightsOutputChannelSize;

        int ohMinusPad = oh - pad;
        int owMinusPad = ow - pad;
        float sum = 0f;

        for (int ic = 0; ic < inputChannels; ic++)
        {
            int inputCIndex = ic * meta.InputChannelSize;
            int weightsInputCIndex = ic * meta.WeightsChannelSize;
            for (int kh = 0; kh < kernelHeight; kh++)
            {
                int ih = kh + ohMinusPad;
                if (ih < 0 || ih >= inputHeight)
                {
                    continue;
                }

                int weightsKernelHIndex = kh * kernelWidth;
                int inputHIndex = ih * inputWidth;
                for (int kw = 0; kw < kernelWidth; kw++)
                {
                    int iw = kw + owMinusPad;
                    if (iw < 0 || iw >= inputWidth)
                    {
                        continue;
                    }

                    float inputVal = input[inputBIndex + inputCIndex + inputHIndex + iw];
                    float weightVal = weights[weightsInputCIndex + weightsOutputCIndex + weightsKernelHIndex + kw];
                    sum += inputVal * weightVal;
                }
            }
        }

        output[outputBIndex + outputCIndex + outputHIndex + ow] = sum;
    }

    /*
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,,,] Convolve2DInputGradient(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int? padding = null)
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
            _accelerator.Allocate3DDenseXY<float>(new Index3D(inputChannels * outputGradientChannels, kernelHeight, kernelWidth));
        using MemoryBuffer3D<float, Stride3D.DenseXY> outputGradientDev =
            _accelerator.Allocate3DDenseXY<float>(new Index3D(batchSize * outputGradientChannels, outputGradientHeight, outputGradientWidth));
        using MemoryBuffer3D<float, Stride3D.DenseXY> inputGradientDev =
            _accelerator.Allocate3DDenseXY<float>(new Index3D(batchSize * inputChannels, inputHeight, inputWidth));

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
            _accelerator.LoadAutoGroupedStreamKernel<
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
        _accelerator.Synchronize();

        CopyInputGrad3DTo4D(inputGradientDev.View, inputGradient);
        return inputGradient;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public override float[,,,] Convolve2DParamGradient(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int? padding = null)
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
            _accelerator.Allocate3DDenseXY<float>(new Index3D(batchSize * inputChannels, inputHeight, inputWidth));
        using MemoryBuffer3D<float, Stride3D.DenseXY> outputGradientDev =
            _accelerator.Allocate3DDenseXY<float>(new Index3D(batchSize * outputGradientChannels, outputGradientHeight, outputGradientWidth));
        using MemoryBuffer3D<float, Stride3D.DenseXY> paramGradientDev =
            _accelerator.Allocate3DDenseXY<float>(new Index3D(inputChannels * outputGradientChannels, kernelHeight, kernelWidth));

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
            _accelerator.LoadAutoGroupedStreamKernel<
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
        _accelerator.Synchronize();

        CopyWeights3DTo4D(paramGradientDev.View, paramGradient);
        return paramGradient;
    }

    #region GPU Kernels - Convolution

    

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
    */

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
            _accelerator.Dispose();
            _context.Dispose();
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
