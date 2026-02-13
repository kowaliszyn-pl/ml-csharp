// Neural Networks in C♯
// File name: DenseLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.Operations.Dropouts;
using NeuralNetworks.Operations.Parameterized;
using NeuralNetworks.ParamInitializers;

namespace NeuralNetworks.Layers;

public class DenseLayer : Layer<float[,], float[,]>
{
    private readonly int _neurons;
    private readonly ActivationFunction<float[,], float[,]> _activationFunction;
    private readonly ParamInitializer _paramInitializer;
    private readonly BaseDropout<float[,]>? _dropout;

    public DenseLayer(int neurons, ActivationFunction<float[,], float[,]> activationFunction, ParamInitializer paramInitializer, BaseDropout<float[,]>? dropout = null)
    {
        _neurons = neurons;
        _activationFunction = activationFunction;
        _paramInitializer = paramInitializer;
        _dropout = dropout;
    }

    public override OperationListBuilder<float[,], float[,]> CreateOperationListBuilder()
    {
        Debug.Assert(Input != null, "Input must not be null here.");

        float[,] weights = _paramInitializer.InitWeights(Input.GetLength(1), _neurons);
        float[] biases = _paramInitializer.InitBiases(_neurons);

        OperationListBuilder<float[,], float[,]> res =
            AddOperation(new WeightMultiply(weights))
            .AddOperation(new BiasAdd(biases))
            .AddOperation(_activationFunction);

        if (_dropout != null)
            res = res.AddOperation(_dropout);

        return res;
    }

    public override string ToString()
        => $"DenseLayer (neurons={_neurons}, activation={_activationFunction}, paramInitializer={_paramInitializer}, dropout={_dropout})";
}
