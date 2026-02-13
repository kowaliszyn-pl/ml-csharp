// Machine Learning Utils
// File name: GlorotInitializer.cs
// Code It Yourself with .NET, 2024

using NeuralNetworks.Core;

using static NeuralNetworks.Core.RandomUtils;

namespace NeuralNetworks.ParamInitializers;

public class GlorotInitializer(SeededRandom? random = null) : RandomInitializer(random)
{
    internal override float[,] InitWeights(int inputColumns, int neurons)
    {
        float stdDev = MathF.Sqrt(2.0f / (inputColumns + neurons));
        return CreateRandomNormal(inputColumns, neurons, Random, 0, stdDev);
    }

    internal override float[,,,] InitWeights(int inputChannels, int outputChannels, int kernelWidth, int kernelHeight)
    {
        float stdDev = MathF.Sqrt(2.0f / (inputChannels + outputChannels));
        return CreateRandomNormal(inputChannels, outputChannels, kernelWidth, kernelHeight, Random, 0, stdDev);
    }

    internal override float[] InitBiases(int neurons)
        => new float[neurons];

    public override string ToString() 
        => $"GlorotInitializer (seed={Seed})";
}