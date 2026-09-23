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

public class DenseLayer(int neurons, ActivationFunction<float[,], float[,]> activationFunction, ParamInitializer paramInitializer, BaseDropout<float[,]>? dropout = null) : Layer<float[,], float[,]>
{
    public override OperationListBuilder<float[,], float[,]> CreateOperationListBuilder()
    {
        Debug.Assert(Input != null, "Input must not be null here.");

        float[,] weights = paramInitializer.InitWeights(Input.GetLength(1), neurons);
        float[] biases = paramInitializer.InitBiases(neurons);

        OperationListBuilder<float[,], float[,]> res =
            AddOperation(new WeightMultiply(weights))
            .AddOperation(new BiasAdd(biases))
            .AddOperation(activationFunction);

        if (dropout != null)
            res = res.AddOperation(dropout);

        return res;
    }

    public ActivationFunction<float[,], float[,]> GetActivationFunction() 
        => activationFunction;

    public override string ToString()
        => $"DenseLayer (neurons={neurons}, activation={activationFunction}, paramInitializer={paramInitializer}, dropout={dropout})";
}
