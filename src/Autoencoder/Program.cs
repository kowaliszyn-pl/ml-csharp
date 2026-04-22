// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.ParamInitializers;

namespace Autoencoder;

internal class AutoencoderModel(int bottleneckDim, SeededRandom? random)
    : BaseModel<float[,,,], float[,,,]>(new MeanSquaredErrorLoss4D(), random)
{
    private const int InnerChannels = 7;
    private const int ImageInnerSize = 28;
    private Layer<float[,], float[,]>? _bottleneckLayer;

    protected override LayerListBuilder<float[,,,], float[,,,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);

        _bottleneckLayer = new DenseLayer(bottleneckDim, new Linear(), initializer);

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
            .AddLayer(new DenseLayer(ImageInnerSize * ImageInnerSize * InnerChannels, new Linear(), initializer))
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

    public float[,] GetBottleneckData()
    {
        return _bottleneckLayer?.Output
            ?? throw new InvalidOperationException("Bottleneck layer output is not available.");
    }

}

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
    }
}
