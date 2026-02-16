// Neural Networks in C♯
// File name: Convolve2DOutputMeta.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Core.Operations.GpuStructs;

public readonly record struct Convolve2DOutputMeta(
    int InputChannels,
    int InputHeight,
    int InputWidth,
    int KernelHeight,
    int KernelWidth,
    int OutputChannels,
    int OutputWidth,
    int InputBatchSize,
    int InputChannelSize,
    int WeightsChannelSize,
    int WeightsOutputChannelSize,
    int OutputBatchSize,
    int OutputChannelSize,
    int PaddingHeight,
    int PaddingWidth,
    int StrideHeight,
    int StrideWidth,
    int DilatationHeight,
    int DilatationWidth
);
