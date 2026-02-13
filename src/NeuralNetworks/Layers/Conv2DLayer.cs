// Machine Learning Utils
// File name: Conv2D.cs
// Code It Yourself with .NET, 2024

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.Operations.Dropouts;
using NeuralNetworks.Operations.Parameterized;
using NeuralNetworks.ParamInitializers;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Layers;

/*
 * TIn and TOut are 4D arrays with the following dimensions: [batch, channels, height, width]
 * TODO: strides, padding, dilation
 */
public class Conv2DLayer : Layer<float[,,,], float[,,,]>
{
    private readonly int _filters;
    private readonly int _kernelSize;
    private readonly ActivationFunction<float[,,,], float[,,,]> _activationFunction;
    private readonly ParamInitializer _paramInitializer;
    private readonly Dropout4D? _dropout;

    public Conv2DLayer(int filters, int kernelSize, ActivationFunction<float[,,,], float[,,,]> activationFunction, ParamInitializer paramInitializer, Dropout4D? dropout = null)
    {
        _filters = filters;
        _kernelSize = kernelSize;
        _activationFunction = activationFunction;
        _paramInitializer = paramInitializer;
        _dropout = dropout;
    }

    public override OperationListBuilder<float[,,,], float[,,,]> CreateOperationListBuilder()
    {
        float[,,,] weights = _paramInitializer.InitWeights(Input!.GetLength(1 /* channels */), _filters, _kernelSize);

        OperationListBuilder<float[,,,], float[,,,]> res = 
            AddOperation(new Conv2D(weights))
            // Add Bias4D
            .AddOperation(_activationFunction);

        if (_dropout != null)
            res = res.AddOperation(_dropout);

        return res;
    }

    protected override void EnsureSameShapeForInput(float[,,,]? input, float[,,,]? inputGradient)
        => EnsureSameShape(input, inputGradient);

    protected override void EnsureSameShapeForOutput(float[,,,]? output, float[,,,]? outputGradient)
        => EnsureSameShape(output, outputGradient);

    public override string ToString()
        => $"Conv2DLayer (filters={_filters}, kernelSize={_kernelSize}, activation={_activationFunction}, paramInitializer={_paramInitializer}, dropout={_dropout})";
}
