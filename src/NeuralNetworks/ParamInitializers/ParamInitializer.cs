// Machine Learning Utils
// File name: ParamInitializer.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.ParamInitializers;

public abstract class ParamInitializer
{
    internal abstract float[] InitBiases(int neurons);
    internal abstract float[,] InitWeights(int inputColumns, int neurons);

    /// <param name="kernelSize">kernelSize is both the height and width of the _weightMultiplyInputGradientKernel</param>
    internal abstract float[,,,] InitWeights(int inputChannels, int outputChannels, int kernelSize);
}
