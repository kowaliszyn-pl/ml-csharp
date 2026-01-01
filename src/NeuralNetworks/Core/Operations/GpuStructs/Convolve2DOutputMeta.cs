// Neural Networks in C♯
// File name: Convolve2DOutputMeta.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Core.Operations.GpuStructs;

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
        //int batchSize,
        //int outputHeight,
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
        //BatchSize = batchSize;
        //OutputHeight = outputHeight;
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
    //public int BatchSize { get; }
    //public int OutputHeight { get; }
    public int OutputWidth { get; }
    public int InputBatchSize { get; }
    public int InputChannelSize { get; }
    public int WeightsChannelSize { get; }
    public int WeightsOutputChannelSize { get; }
    public int OutputBatchSize { get; }
    public int OutputChannelSize { get; }
}
