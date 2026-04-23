// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.ParamInitializers;

namespace Autoencoder;

internal class AutoencoderModel : BaseModel<float[,,,], float[,,,]>
{

    public AutoencoderModel(int bottleneckDim, SeededRandom random): base(random)
    {
        _bottleneckDim = bottleneckDim;
    }

    public AutoencoderModel(string modelFilePath) : base(modelFilePath)
    {
        _bottleneckDim = GetEncodedRepresentation().GetLength(1);
    }

    private const int InnerChannels = 7;
    private const int ImageInnerSize = 28;
    private readonly int _bottleneckDim;
    private Layer<float[,], float[,]>? _bottleneckLayer;
    private Layer<float[,], float[,]>? _firstDecoderLayer;

    protected override LayerListBuilder<float[,,,], float[,,,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);

        _bottleneckLayer = new DenseLayer(_bottleneckDim, new Linear(), initializer);
        _firstDecoderLayer = new DenseLayer(ImageInnerSize * ImageInnerSize * InnerChannels, new Linear(), initializer);

        return
            AddLayer(new Conv2DLayer(
                kernels: 14,
                kernelHeight: 5,
                kernelWidth: 5,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ))
            .AddLayer(new Conv2DLayer(
                kernels: 7,
                kernelHeight: 5,
                kernelWidth: 5,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ))
            .AddLayer(new FlattenLayer())
            .AddLayer(_bottleneckLayer)
            .AddLayer(_firstDecoderLayer)
            .AddLayer(new UnflattenLayer(InnerChannels, ImageInnerSize, ImageInnerSize))
            .AddLayer(new Conv2DLayer(
                kernels: 14,
                kernelHeight: 5,
                kernelWidth: 5,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ))
            .AddLayer(new Conv2DLayer(
                kernels: 1,
                kernelHeight: 5,
                kernelWidth: 5,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer
            ));
    }

    /// <summary>
    /// Gets the encoded representation (latent data) produced by the bottleneck layer of the model.
    /// </summary>
    /// <returns>
    /// A two-dimensional array of floating-point values representing the output of the bottleneck layer.
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown if the bottleneck layer output is not available.</exception>
    public float[,] GetEncodedRepresentation()
    {
        return _bottleneckLayer?.Output
            ?? throw new InvalidOperationException("Bottleneck layer output is not available.");
    }

    /// <summary>
    /// Forward encoded representation and return the decoded output. This can be used to visualize the output of the
    /// decoder part of the autoencoder based on randomly generated encoded data or to see how the decoder reconstructs
    /// the input data from the encoded (bottleneck) representation.
    /// </summary>
    public float[,,,] Decode(float[,] encoded)
    {
        // We need to pass the encoded data through the first decoder layer and then through the remaining layers of the model.

        if (_firstDecoderLayer is null)
            throw new InvalidOperationException("Decoder layer is not initialized.");

        return InferenceFromLayer(_firstDecoderLayer, encoded);
    }
}

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
    }
}
