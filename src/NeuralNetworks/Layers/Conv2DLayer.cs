// Neural Networks in C♯
// File name: Conv2DLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.Operations.Dropouts;
using NeuralNetworks.Operations.Parameterized;
using NeuralNetworks.ParamInitializers;

namespace NeuralNetworks.Layers;

/*
 * TIn and TOut are 4D arrays with the following dimensions: [batch, channels, height, width]
 */
public class Conv2DLayer(
    int kernels,
    int kernelHeight, int kernelWidth,
    ActivationFunction<float[,,,], float[,,,]> activationFunction,
    ParamInitializer paramInitializer,
    Dropout4D? dropout = null,
    bool addBias = false,
    int? paddingHeight = null, int? paddingWidth = null,
    int strideHeight = 1, int strideWidth = 1,
    int dilatationHeight = 0, int dilatationWidth = 0
) : Layer<float[,,,], float[,,,]>
{
    public override OperationListBuilder<float[,,,], float[,,,]> CreateOperationListBuilder()
    {
        int inputChannels = Input!.GetLength(1 /* channels */);
        float[,,,] weights = paramInitializer.InitWeights(inputChannels, kernels, kernelHeight, kernelWidth);

        OperationListBuilder<float[,,,], float[,,,]> res =
            AddOperation(new Conv2D(weights, paddingHeight ?? kernelHeight / 2, paddingWidth ?? kernelWidth / 2, strideHeight, strideWidth, dilatationHeight, dilatationWidth));

        if (addBias)
        {
            // [batch = 1, kernels, outputLength]
            float[] bias = paramInitializer.InitBiases(kernels);
            //res.AddOperation(new BiasAddConv1D(bias));
        }

        res.AddOperation(activationFunction);

        if (dropout != null)
            res = res.AddOperation(dropout);

        return res;
    }

    public override string ToString()
        => $"Conv2DLayer (kernels={kernels}, kernelHeight={kernelHeight}, kernelWidth={kernelWidth}, activation={activationFunction}, paramInitializer={paramInitializer}, dropout={dropout}, addBias={addBias}, paddingHeight={paddingHeight}, paddingWidth={paddingWidth}, strideHeight={strideHeight}, strideWidth={strideWidth}, dilatationHeight={dilatationHeight}, dilatationWidth={dilatationWidth})";
}
