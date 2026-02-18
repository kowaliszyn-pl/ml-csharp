// Neural Networks in C♯
// File name: 
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.ParamInitializers;

namespace NeuralNetworksExamples;

internal class Ecg200Model(SeededRandom? random)
    : BaseModel<float[,,], float[,]>(new SoftmaxCrossEntropyLoss(), random)
{
    protected override LayerListBuilder<float[,,], float[,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);
        return
            AddLayer(new Conv1DLayer(
                kernels: 16,
                kernelLength: 5,
                stride: 1,
                activationFunction: new ReLU3D(),
                paramInitializer: initializer
            ))
            .AddLayer(new MaxPooling3DLayer(2))
            .AddLayer(new Conv1DLayer(
                kernels: 32,
                kernelLength: 3,
                stride: 1,
                activationFunction: new ReLU3D(),
                paramInitializer: initializer
            ))
            .AddLayer(new GlobalAveragePooling3DLayer())
            .AddLayer(new DenseLayer(1, new Sigmoid(), initializer));
    }
}

internal class Ecg200
{
    internal static void Run()
    {

    }
}