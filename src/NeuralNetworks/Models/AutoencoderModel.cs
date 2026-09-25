// Neural Networks in C♯
// File name: AutoencoderModel.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Operations;

namespace NeuralNetworks.Models;

public abstract class AutoencoderModel<T>(
    Loss<T>? defaultLossFunction,
    SeededRandom? random,
    string? modelFilePath)

    : BaseModel<T, T>(
        defaultLossFunction,
        random,
        modelFilePath)

    where T : notnull
{
    /// <summary>
    /// Gets or sets the bottleneck layer of the autoencoder model, which represents the compressed latent representation of the input data.
    /// </summary>
    protected DenseLayer? BottleneckLayer { get; set; }

    /// <summary>
    /// Gets or sets the first decoder layer of the autoencoder model, which is responsible for reconstructing the input data from the encoded representation.
    /// </summary>
    protected DenseLayer? FirstDecoderLayer { get; set; }

    /// <summary>
    /// Gets the encoded representation (latent data) produced by the bottleneck layer of the model.
    /// </summary>
    /// <returns>
    /// A two-dimensional array of floating-point values representing the output of the bottleneck layer.
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown if the bottleneck layer output is not available.</exception>
    public float[,] GetEncodedRepresentation()
    {
        return BottleneckLayer?.Output
            ?? throw new InvalidOperationException("Bottleneck layer output is not available.");
    }

    /// <summary>
    /// Forward encoded representation and return the decoded output. This can be used to visualize the output of the
    /// decoder part of the autoencoder based on randomly generated encoded data or to see how the decoder reconstructs
    /// the input data from the encoded (bottleneck) representation.
    /// </summary>
    public T Decode(float[,] encoded)
    {
        // We need to pass the encoded data through the first decoder layer and then through the remaining layers of the model.

        return FirstDecoderLayer is null
            ? throw new InvalidOperationException("Decoder layer is not initialized.")
            : InferFromLayer(FirstDecoderLayer, encoded);
    }

    public Operation GetBottleneckActivationFunction()
    {
        return BottleneckLayer?.GetActivationFunction()
            ?? throw new InvalidOperationException("Bottleneck layer is not initialized.");
    }
}
