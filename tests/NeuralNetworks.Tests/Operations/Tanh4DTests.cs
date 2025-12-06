// Machine Learning Utils
// File name: Tanh4DTests.cs
// Code It Yourself with .NET, 2024

namespace NeuralNetworks.Operations.Tests;

[TestClass]
public class Tanh4DTests
{
    [TestMethod]
    public void CalcOutputTest()
    {
        // Arrange
        Tanh4D tanh4D = new();
        float[,,,] input = new float[,,,] {
            {
                {
                    { 0, 1 },
                    { -1, 2 }
                },
                {
                    { -2, 3 },
                    { -3, 4 }
                }
            },
            {
                {
                    { 5, -4 },
                    { 6, -5 }
                },
                {
                    { 7, -6 },
                    { 8, -7 }
                }
            }
        };

        // Act
        float[,,,] output = tanh4D.Forward(input, true);

        // Assert
        Assert.AreEqual(Math.Tanh(0), output[0, 0, 0, 0], 1e-6);
        Assert.AreEqual(Math.Tanh(1), output[0, 0, 0, 1], 1e-6);
        Assert.AreEqual(Math.Tanh(-1), output[0, 0, 1, 0], 1e-6);
        Assert.AreEqual(Math.Tanh(2), output[0, 0, 1, 1], 1e-6);
        Assert.AreEqual(Math.Tanh(-2), output[0, 1, 0, 0], 1e-6);
        Assert.AreEqual(Math.Tanh(3), output[0, 1, 0, 1], 1e-6);
        Assert.AreEqual(Math.Tanh(-3), output[0, 1, 1, 0], 1e-6);
        Assert.AreEqual(Math.Tanh(4), output[0, 1, 1, 1], 1e-6);
        Assert.AreEqual(Math.Tanh(5), output[1, 0, 0, 0], 1e-6);
        Assert.AreEqual(Math.Tanh(-4), output[1, 0, 0, 1], 1e-6);
        Assert.AreEqual(Math.Tanh(6), output[1, 0, 1, 0], 1e-6);
        Assert.AreEqual(Math.Tanh(-5), output[1, 0, 1, 1], 1e-6);
        Assert.AreEqual(Math.Tanh(7), output[1, 1, 0, 0], 1e-6);
        Assert.AreEqual(Math.Tanh(-6), output[1, 1, 0, 1], 1e-6);
        Assert.AreEqual(Math.Tanh(8), output[1, 1, 1, 0], 1e-6);
        Assert.AreEqual(Math.Tanh(-7), output[1, 1, 1, 1], 1e-6);
    }

    [TestMethod]
    public void CalcInputGradientTest()
    {
        // Arrange
        Tanh4D tanh4D = new();
        float[,,,] input = new float[,,,] {
            {
                {
                    { 0, 1 },
                    { -1, 2 }
                },
                {
                    { -2, 3 },
                    { -3, 4 }
                }
            },
            {
                {
                    { 5, -4 },
                    { 6, -5 }
                },
                {
                    { 7, -6 },
                    { 8, -7 }
                }
            }
        };
        float[,,,] outputGradient = new float[,,,] {
            {
                {
                    { 1, 1 },
                    { 1, 1 }
                },
                {
                    { 1, 1 },
                    { 1, 1 }
                }
            },
            {
                {
                    { 1, 1 },
                    { 1, 1 }
                },
                {
                    { 1, 1 },
                    { 1, 1 }
                }
            }
        };
        _ = tanh4D.Forward(input, true);

        // Act
        float[,,,] inputGradient = tanh4D.Backward(outputGradient);

        // Assert
        for (int i = 0; i < input.GetLength(0); i++)
        {
            for (int j = 0; j < input.GetLength(1); j++)
            {
                for (int k = 0; k < input.GetLength(2); k++)
                {
                    for (int l = 0; l < input.GetLength(3); l++)
                    {
                        float expectedGradient = outputGradient[i, j, k, l] * (1 - (float)Math.Pow(Math.Tanh(input[i, j, k, l]), 2));
                        Assert.AreEqual(expectedGradient, inputGradient[i, j, k, l], 1e-6);
                    }
                }
            }
        }
    }
}
