// Neural Networks in C♯
// File name: DenseLayer.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Operations;
using NeuralNetworks.ParamInitializers;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Layers;

public class DenseLayer : Layer<float[,], float[,]>
{
    private readonly int _neurons;
    private readonly Operation2D _activationFunction;
    private readonly ParamInitializer _paramInitializer;
    private readonly Dropout2D? _dropout;

    public DenseLayer(int neurons, Operation2D activationFunction, ParamInitializer paramInitializer, Dropout2D? dropout = null)
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

    protected override void EnsureSameShapeForInput(float[,]? input, float[,]? inputGradient)
        => EnsureSameShape(input, inputGradient);

    protected override void EnsureSameShapeForOutput(float[,]? output, float[,]? outputGradient)
        => EnsureSameShape(output, outputGradient);

    public override string ToString()
        => $"DenseLayer (neurons={_neurons}, activation={_activationFunction}, paramInitializer={_paramInitializer}, dropout={_dropout})";
}
