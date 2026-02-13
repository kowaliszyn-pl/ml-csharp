// Machine Learning Utils
// File name: ParamInitializer.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.ParamInitializers;

public abstract class ParamInitializer
{
    /// <summary>
    /// Initializes an array of bias values for a neural network layer.
    /// </summary>
    /// <remarks>Derived classes should implement this method to provide specific bias initialization logic
    /// appropriate for the neural network architecture or training strategy.</remarks>
    /// <param name="size">The number of bias values to initialize (neurons, kernels, etc.). Must be a positive integer.</param>
    /// <returns>An array of single-precision floating-point values representing the initialized biases. The length of the array
    /// is equal to the specified size.</returns>
    internal abstract float[] InitBiases(int size);
    
    internal abstract float[,] InitWeights(int inputColumns, int neurons);

    internal abstract float[,,] InitWeights(int inputChannels, int outputChannels, int kernelLength);

    /// <param name="kernelSize">kernelSize is both the height and width of the _weightMultiplyInputGradientKernel</param>
    internal abstract float[,,,] InitWeights(int inputChannels, int outputChannels, int kernelWidth, int kernelHeight);
}
