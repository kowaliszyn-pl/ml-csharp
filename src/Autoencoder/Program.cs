// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.Operations.Dropouts;
using NeuralNetworks.ParamInitializers;

using static System.Console;
using static NeuralNetworks.Core.ArrayUtils;

namespace Autoencoder;

internal class AutoencoderModel(int bottleneckDim, SeededRandom? random)
    : BaseModel<float[,,,], float[,,,]>(new MeanSquaredErrorLoss4D(), random)
{
    Layer<float[,], float[,]>? bottleneckLayer;

    protected override LayerListBuilder<float[,,,], float[,,,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);
        // Dropout4D dropout = new(0.80f, Random);
        Dropout2D dropout = new(0.80f, Random);

        bottleneckLayer = new DenseLayer(bottleneckDim, new Linear(), initializer, dropout);

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
            .AddLayer(bottleneckLayer)
            .AddLayer(new DenseLayer(28 * 28 * 7, new Linear(), initializer))
            .AddLayer(new UnflattenLayer(7, 28, 28))
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
        return bottleneckLayer?.Output
            ?? throw new InvalidOperationException("Bottleneck layer output is not available.");
    }

}

internal class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
    }
}
