// Neural Networks in C♯
// File name: NormallyDistributedRandomInitializer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

using static NeuralNetworks.Core.RandomUtils;

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
