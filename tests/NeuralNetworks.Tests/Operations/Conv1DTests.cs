// Neural Networks in C♯
// File name: Conv1DTests.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

using NeuralNetworks.Operations.Parameterized;

namespace NeuralNetworks.Tests.Operations;

[TestClass]
public class Conv1DTests
{
    [TestMethod]

    public void Conv1DOperationTest()
    {
        // One observation
        float[] obs = [1f, 2f, 3f, 4f, 5f];

        // One filter
        float[] filter = [1f, 0f, -2f];

        // Create input data: 1 observation, 1 channel, 5
        float[,,] input = new float[1, 1, 5];
        for (int i = 0; i < 5; i++)
        {
            input[0, 0, i] = obs[i];
        }

        // Create param data: 1 channel, 1 filter, 3
        float[,,] param = new float[1, 1, 3];
        for (int i = 0; i < 3; i++)
        {
            param[0, 0, i] = filter[i];
        }

        // Create a Conv2DOperation
        Conv1D conv1DOperation = new(param, 1);

        // Do forward
        float[,,] output = conv1DOperation.Forward(input, true);

        // Assert
        Assert.AreEqual(1, output.GetLength(0));
        Assert.AreEqual(1, output.GetLength(1));
        Assert.AreEqual(5, output.GetLength(2));

        CollectionAssert.AreEqual(new float[,,] {
            {
                { -4, 1 - 6, 2 - 8, 3 - 10, 4 }
            }
        }, output);


        // Do backward
        float[,,] outputGradient = new float[1, 1, 5];
        for (int i = 0; i < 5; i++)
        {
            outputGradient[0, 0, i] = -(i + 1) * 2; // -2, -4, -6, -8, -10
        }

        float[,,] inputGradient = conv1DOperation.Backward(outputGradient);

        // Assert
        Assert.AreEqual(1, inputGradient.GetLength(0));
        Assert.AreEqual(1, inputGradient.GetLength(1));
        Assert.AreEqual(5, inputGradient.GetLength(2));


        // The input gradient is obtained by convolving the gradient of the loss with respect to the output (outputGradient) with the reversed filter. Since the filter is [1, 0, -2], the reversed filter is [-2, 0, 1]. The convolution is done with 'same' padding, which means that the output has the same length as the input. The convolution is done by sliding the reversed filter over the output gradient and calculating the dot product at each position.
        CollectionAssert.AreEqual(new float[,,] {
            {
                {
                    (-2) * 0 /* padding */ + 0 * (-2) + 1 * (-4), // -4
                    (-2) * (-2) + 0 * (-4) + 1 * (-6), // -2
                    (-2) * (-4) + 0 * (-6) + 1 * (-8), // 0
                    (-2) * (-6) + 0 * (-8) + 1 * (-10), // 2
                    (-2) * (-8) + 0 * (-10) + 1 * 0 /* padding */ // 16
                }
            }
        }, inputGradient);

        // Check param gradient
        float[,,] paramGradient = conv1DOperation.ParamGradient;

        // Assert
        Assert.AreEqual(1, paramGradient.GetLength(0));
        Assert.AreEqual(1, paramGradient.GetLength(1));
        Assert.AreEqual(3, paramGradient.GetLength(2));

        // The gradient of the weights is obtained by convolving the input with the gradient of the loss with respect to the output (outputGradient).
        CollectionAssert.AreEqual(new float[,,] {
            {
                {
                    (-2) * 0 /* padding */  + (-4) * 1 + (-6) * 2 + (-8) * 3 + (-10) * 4,
                    (-2) * 1  + (-4) * 2 + (-6) * 3 + (-8) * 4 + (-10) * 5,
                    (-2) * 2  + (-4) * 3 + (-6) * 4 + (-8) * 5 + (-10) * 0 /* padding */
                }
            }
        }, paramGradient);

    }
}
