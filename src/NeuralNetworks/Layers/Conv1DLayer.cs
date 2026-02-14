// Neural Networks in C♯
// File name: Conv1DLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.Operations.Dropouts;
using NeuralNetworks.Operations.Parameterized;
using NeuralNetworks.ParamInitializers;

namespace NeuralNetworks.Layers;

/// <summary>
/// Represents a one-dimensional convolutional layer that applies multiple kernels to input data, enabling feature
/// extraction in neural network models.
/// </summary>
/// <remarks>
/// This layer is commonly used in sequence modeling tasks, such as time series analysis or natural
/// language processing. The combination of kernels, kernel size, stride, and dilation allows for flexible feature
/// extraction from one-dimensional data.
/// <para/>
/// The input data is [batch, channels, length].
/// The output data is [batch, kernels, length].
/// </remarks>
/// <param name="kernels">The number of convolution kernels to apply. Determines the depth of the output feature map. Must be a positive
/// integer.</param>
/// <param name="kernelLength">The size of the convolution kernel. Specifies the width of each filter applied to the input data. Must be a positive
/// integer.</param>
/// <param name="activationFunction">The activation function to apply after the convolution operation. Introduces non-linearity to the layer's output.</param>
/// <param name="paramInitializer">The initializer used to set the initial values of the layer's weights. Influences the starting point for training.</param>
/// <param name="dropout">An optional dropout layer applied during training to reduce overfitting by randomly setting a fraction of input
/// units to zero.</param>
/// <param name="padding">The amount of zero-padding to add to the input. If not specified, the layer will use half of the kernel size as padding, ensuring that the output has the same width as the input. Must be a non-negative integer.</param>
/// <param name="stride">The stride of the convolution operation. Defines how many input positions the filter moves at each step. Must be a
/// positive integer.</param>
/// <param name="dilatation">The dilation rate for the convolution. Expands the kernel by inserting spaces between elements, allowing for larger
/// receptive fields.</param>
public class Conv1DLayer(
    int kernels,
    int kernelLength,
    ActivationFunction<float[,,], float[,,]> activationFunction,
    ParamInitializer paramInitializer,
    Dropout3D? dropout = null,
    bool addBias = false,
    int? padding = null,
    int stride = 1,
    int dilatation = 0) : Layer<float[,,], float[,,]>
{
    public override OperationListBuilder<float[,,], float[,,]> CreateOperationListBuilder()
    {
        float[,,] weights = paramInitializer.InitWeights(Input!.GetLength(1 /* channels */), kernels, kernelLength);
        

        OperationListBuilder<float[,,], float[,,]> res = 
            AddOperation(new Conv1D(weights, padding ?? kernelLength / 2, stride, dilatation));

        if (addBias)
        {
            // [batch = 1, kernels, outputLength]
            float[] bias = paramInitializer.InitBiases(kernels);
            res.AddOperation(new BiasAddConv1D(bias));
        }

        if (dropout is not null)
            res.AddOperation(dropout);
        res.AddOperation(activationFunction);
        return res;
    }

    public override string ToString() 
        => $"Conv1DLayer (kernels={kernels}, kernelLength={kernelLength}, activation={activationFunction}, paramInitializer={paramInitializer}, dropout={dropout}, addBias={addBias}, padding={padding}, stride={stride}, dilatation={dilatation})";
}
