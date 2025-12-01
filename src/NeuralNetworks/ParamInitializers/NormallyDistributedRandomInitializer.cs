// Machine Learning Utils
// File name: NormallyDistributedRandomInitializer.cs
// Code It Yourself with .NET, 2024

using NeuralNetworks.Core;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.ParamInitializers;

public class NormallyDistributedRandomInitializer(float mean = 0, float stdDev = 1, SeededRandom? random = null) : RandomInitializer(random)
{
    internal override float[] InitBiases(int neurons) 
        => CreateRandomNormal(neurons, Random, mean, stdDev);

    internal override float[,] InitWeights(int inputColumns, int neurons) 
        => CreateRandomNormal(inputColumns, neurons, Random, mean, stdDev);

    public override string ToString() 
        => $"NormallyDistributedRandomInitializer (seed={Seed}, mean={mean}, stdDev={stdDev})";
}
