// Machine Learning Utils
// File name: Conv2DOperationTests.cs
// Code It Yourself with .NET, 2024

using NeuralNetworks.Operations.Parameterized;

namespace NeuralNetworks.Operations.Tests;

// TODO: compare the test results with
// https://github.com/mashmawy/ConvNetCS/blob/master/ConvNetCS/Network/Layers/ConvLayer.cs
// https://github.com/cbovar/ConvNetSharp/blob/master/src/ConvNetSharp.Core/Layers/ConvLayer.cs
// https://github.com/romanoza/NeuroSharp/blob/main/NeuroSharp/Layers/ConvolutionalLayer.cs

[TestClass]
public class Conv2DTests
{
    [TestMethod]

    public void Conv2DOperationTest()
    {
        // One observation
        float[,] obs = new float[,] {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        };

        // One filter
        float[,] filter = new float[,] {
            { 1, 0, 1 },
            { 0, 1, 0 },
            { 1, 0, 1 }
        };

        // Create input data: 1 observation, 1 channel, 3x3
        float[,,,] input = new float[1, 1, 3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                input[0, 0, i, j] = obs[i, j];
            }
        }

        // Create param data: 1 channel, 1 filter, 3x3
        float[,,,] param = new float[1, 1, 3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                param[0, 0, i, j] = filter[i, j];
            }
        }

        // Create a Conv2DOperation
        Conv2D conv2DOperation = new(param, 1, 1);

        // Do forward
        float[,,,] output = conv2DOperation.Forward(input, true);

        // Assert
        Assert.AreEqual(1, output.GetLength(0));
        Assert.AreEqual(1, output.GetLength(1));
        Assert.AreEqual(3, output.GetLength(2));
        Assert.AreEqual(3, output.GetLength(3));

        CollectionAssert.AreEqual(new float[,,,] {
            {
                {
                    { 1 + 5,       2 + 4 + 6,            3 + 5 },
                    { 2 + 4 + 8, 1 + 3 + 5 + 7 + 9, 2 + 6 + 8 },
                    { 5 + 7,       4 + 6 + 8,            5 + 9 }
                }
            }
        }, output);

        // Do backward
        float[,,,] outputGradient = new float[1, 1, 3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                outputGradient[0, 0, i, j] = 1;
            }
        }

        float[,,,] inputGradient = conv2DOperation.Backward(outputGradient);

        // Assert
        Assert.AreEqual(1, inputGradient.GetLength(0));
        Assert.AreEqual(1, inputGradient.GetLength(1));
        Assert.AreEqual(3, inputGradient.GetLength(2));
        Assert.AreEqual(3, inputGradient.GetLength(3));

        // The input gradient is obtained by convolving the gradient of the loss with respect to the output (outputGradient) with the flipped convolutional kernel (param).
        CollectionAssert.AreEqual(new float[,,,] {
            {
                {
                    { 1 + 1,       1 + 1 + 1,            1 + 1 },
                    { 1 + 1 + 1, 1 + 1 + 1 + 1 + 1, 1 + 1 + 1 },
                    { 1 + 1,       1 + 1 + 1,             1 + 1 }
                }
            }
        }, inputGradient);

        // Check param gradient
        float[,,,] paramGradient = conv2DOperation.ParamGradient;

        // Assert
        Assert.AreEqual(1, paramGradient.GetLength(0));
        Assert.AreEqual(1, paramGradient.GetLength(1));
        Assert.AreEqual(3, paramGradient.GetLength(2));
        Assert.AreEqual(3, paramGradient.GetLength(3));

        // The gradient of the weights is obtained by convolving the input with the gradient of the loss with respect to the output (outputGradient).
        CollectionAssert.AreEqual(new float[,,,] {
            {
                {
                    { 1 + 2 + 4 + 5,             1 + 2 + 3 + 4 + 5 + 6,                  2 + 3 + 5 + 6 },
                    { 1 + 2 + 4 + 5 + 7 + 8, 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, 2 + 3 + 5 + 6 + 8 + 9 },
                    { 4 + 5 + 7 + 8,             4 + 5 + 6 + 7 + 8 + 9,                  5 + 6 + 8 + 9 }
                }
            }
        }, paramGradient);
    }

    [TestMethod]

    public void Conv2DOperationOnAsymmetricFilterTest()
    {
        // One observation
        float[,] obs = new float[,] {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        };

        // One filter
        float[,] filter = new float[,] {
            { 1, 0, 0 },
            { 1, 0, 0 },
            { 1, 0, 0 }
        };

        // Create input data: 1 observation, 1 channel, 3x3
        float[,,,] input = new float[1, 1, 3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                input[0, 0, i, j] = obs[i, j];
            }
        }

        // Create param data: 1 channel, 1 filter, 3x3
        float[,,,] param = new float[1, 1, 3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                param[0, 0, i, j] = filter[i, j];
            }
        }

        // Create a Conv2DOperation
        Conv2D conv2DOperation = new(param, 1, 1);

        // Do forward
        float[,,,] output = conv2DOperation.Forward(input, true);

        // Assert
        Assert.AreEqual(1, output.GetLength(0));
        Assert.AreEqual(1, output.GetLength(1));
        Assert.AreEqual(3, output.GetLength(2));
        Assert.AreEqual(3, output.GetLength(3));

        CollectionAssert.AreEqual(new float[,,,] {
            {
                {
                    { 0, 5, 7 },
                    { 0, 12, 15 },
                    { 0, 11, 13 }
                }
            }
        }, output);

        // Do backward
        float[,,,] outputGradient = new float[1, 1, 3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                outputGradient[0, 0, i, j] = 1;
            }
        }

        float[,,,] inputGradient = conv2DOperation.Backward(outputGradient);

        // Assert
        Assert.AreEqual(1, inputGradient.GetLength(0));
        Assert.AreEqual(1, inputGradient.GetLength(1));
        Assert.AreEqual(3, inputGradient.GetLength(2));
        Assert.AreEqual(3, inputGradient.GetLength(3));

        // The input gradient is obtained by convolving the gradient of the loss with respect to the output (outputGradient) with the flipped (left-to-right and top-to-bottom) convolutional kernel (param).
        CollectionAssert.AreEqual(new float[,,,] {
            {
                {
                    { 1 + 1,       1 + 1, 0 },
                    { 1 + 1 + 1, 1 + 1 + 1, 0 },
                    { 1 + 1,       1 + 1, 0 }
                }
            }
        }, inputGradient);

        // Check param gradient
        float[,,,] paramGradient = conv2DOperation.ParamGradient;

        // Assert
        Assert.AreEqual(1, paramGradient.GetLength(0));
        Assert.AreEqual(1, paramGradient.GetLength(1));
        Assert.AreEqual(3, paramGradient.GetLength(2));
        Assert.AreEqual(3, paramGradient.GetLength(3));

        // The gradient of the weights is obtained by convolving the input with the gradient of the loss with respect to the output (outputGradient).
        CollectionAssert.AreEqual(new float[,,,] {
            {
                {
                    { 1 + 2 + 4 + 5,             1 + 2 + 3 + 4 + 5 + 6,                  2 + 3 + 5 + 6 },
                    { 1 + 2 + 4 + 5 + 7 + 8, 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9, 2 + 3 + 5 + 6 + 8 + 9 },
                    { 4 + 5 + 7 + 8,             4 + 5 + 6 + 7 + 8 + 9,                  5 + 6 + 8 + 9 }
                }
            }
        }, paramGradient);
    }

    [TestMethod]
    public void Conv2DOperationOnMultipleChannelsTest()
    {
        const int inputChannels = 2;
        const int outputChannels = 3;
        const int height = 3;
        const int width = 3;

        // One observation
        float[,] obs = new float[,] {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        };

        // One filter
        float[,] filter = new float[,] {
            { 1, 0, 0 },
            { 1, 0, 0 },
            { 1, 0, 0 }
        };

        // Create input data: 1 observation, 2 channels, 3x3
        float[,,,] input = new float[1, inputChannels, height, width];
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                for (int ic = 0; ic < inputChannels; ic++)
                {
                    input[0, ic, h, w] = obs[h, w];
                    input[0, ic, h, w] = obs[h, w];
                }
            }
        }

        // Create param data: 2 channels, 3 filters, 3x3
        float[,,,] param = new float[inputChannels, outputChannels, height, width];
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                for (int oc = 0; oc < outputChannels; oc++)
                {
                    param[0, oc, h, w] = filter[h, w];
                    param[1, oc, h, w] = filter[h, w];
                }
            }
        }

        // Create a Conv2DOperation
        Conv2D conv2DOperation = new(param, 1, 1);

        // Do forward
        float[,,,] output = conv2DOperation.Forward(input, true);

        // Assert
        Assert.AreEqual(1, output.GetLength(0));
        Assert.AreEqual(outputChannels, output.GetLength(1));
        Assert.AreEqual(height, output.GetLength(2));
        Assert.AreEqual(width, output.GetLength(3));
    }
}
