// Neural Networks in C♯
// File name: RangeInitializer.cs
// www.kowaliszyn.pl, 2025

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.ParamInitializers;

public class RangeInitializer(float from, float to) : ParamInitializer
{
    internal override float[] InitBiases(int neurons)
        => new float[neurons];

    internal override float[,] InitWeights(int inputColumns, int neurons)
        => CreateRange(inputColumns, neurons, from, to);

    internal override float[,,,] InitWeights(int inputChannels, int outputChannels, int kernelWidth, int kernelHeight)
       => CreateRange(inputChannels, outputChannels, kernelWidth, kernelHeight, from, to);

    internal override float[,,] InitWeights(int inputChannels, int outputChannels, int kernelLength) => throw new NotImplementedException();

    public override string ToString() => $"RangeInitializer (from={from}, to={to})";
    
}
