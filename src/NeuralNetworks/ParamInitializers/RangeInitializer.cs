// Machine Learning Utils
// File name: RangeInitializer.cs
// Code It Yourself with .NET, 2024

using System;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.ParamInitializers;

public class RangeInitializer(float from, float to) : ParamInitializer
{
    internal override float[] InitBiases(int neurons) 
        => CreateZeros(neurons);

    internal override float[,] InitWeights(int inputColumns, int neurons) 
        => CreateRange(inputColumns, neurons, from, to);

    internal override float[,,,] InitWeights(int inputChannels, int outputChannels, int kernelSize)
       => CreateRange(inputChannels, outputChannels, kernelSize, kernelSize, from, to);

    public override string ToString() => $"RangeInitializer (from={from}, to={to})";
    
}
