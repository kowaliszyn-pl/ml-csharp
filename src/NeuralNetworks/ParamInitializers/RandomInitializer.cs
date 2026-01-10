// Neural Networks in C♯
// File name: RandomInitializer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

using static NeuralNetworks.Core.RandomUtils;

namespace NeuralNetworks.ParamInitializers;

public class RandomInitializer : ParamInitializer
{
    private readonly Random _random;
    private readonly int? _seed;

    public RandomInitializer(SeededRandom? random = null)
    {
        if (random != null)
        {
            _random = random;
            _seed = random.Seed;
        }
        else
        {
            _random = new Random();
            _seed = null;
        }
    }

    protected Random Random => _random;

    protected int? Seed => _seed;

    internal override float[] InitBiases(int neurons)
        => CreateRandom(neurons, _random);

    internal override float[,] InitWeights(int inputColumns, int neurons)
        => CreateRandom(inputColumns, neurons, _random);

    internal override float[,,,] InitWeights(int inputChannels, int outputChannels, int kernelSize)
        => CreateRandom(inputChannels, outputChannels, kernelSize, kernelSize, _random);

    public override string ToString() => $"RandomInitializer (seed={_seed})";

}
