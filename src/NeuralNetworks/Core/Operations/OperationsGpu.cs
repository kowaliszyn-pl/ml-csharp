// Neural Networks in C♯
// File name: OperationsGpu.cs
// www.kowaliszyn.pl, 2025

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

namespace NeuralNetworks.Core.Operations;

internal class OperationsGpu: OperationsSpan, IDisposable
{
    private static readonly Context s_context;
    private static readonly Accelerator s_accelerator;
    private bool _disposedValue;

    public OperationsGpu()
    {
        // Initialize ILGPU context and accelerator
        s_context = Context.Create(builder => builder.Cuda().CPU());
        s_accelerator = s_context.GetPreferredDevice(preferCPU: false).CreateAccelerator(s_context);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float[,] WeightMultiplyCalcOutput(float[,] input, float[,] weights)
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float[,] WeightMultiplyCalcInputGradient(float[,] outputGradient, float[,] weights)
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float[,] WeightMultiplyCalcParamGradient(float[,] input, float[,] outputGradient)
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

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            if (disposing)
            {
                // TODO: dispose managed state (managed objects)
            }

            // TODO: free unmanaged resources (unmanaged objects) and override finalizer
            // TODO: set large fields to null
            _disposedValue = true;
        }
    }

    // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
    // ~OperationsGpu()
    // {
    //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
    //     Dispose(disposing: false);
    // }

    void IDisposable.Dispose()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}
