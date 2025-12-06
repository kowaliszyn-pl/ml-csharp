// Machine Learning Utils
// File name: Tanh2DTests.cs
// Code It Yourself with .NET, 2024


namespace NeuralNetworks.Operations.Tests;

[TestClass]
public class Tanh2DTests
{
    [TestMethod]
    public void CalcOutputTest()
    {
        // Arrange
        Tanh2D tanh2D = new();
        float[,] input = new float[,] { { 0, 1 }, { -1, 2 } };

        // Act
        float[,] output = tanh2D.Forward(input, true);

        // Assert
        Assert.AreEqual(Math.Tanh(0), output[0, 0], 1e-6);
        Assert.AreEqual(Math.Tanh(1), output[0, 1], 1e-6);
        Assert.AreEqual(Math.Tanh(-1), output[1, 0], 1e-6);
        Assert.AreEqual(Math.Tanh(2), output[1, 1], 1e-6);
    }

    [TestMethod]
    public void CalcInputGradientTest()
    {
        // Arrange
        Tanh2D tanh2D = new();
        float[,] input = new float[,] { { 0, 1 }, { -1, 2 } };
        _ = tanh2D.Forward(input, true);
        float[,] outputGradient = new float[,] { { 1, 1 }, { 1, 1 } };

        // Act
        float[,] inputGradient = tanh2D.Backward(outputGradient);

        // Assert
        Assert.AreEqual(1 * (1 - Math.Pow(Math.Tanh(0), 2)), inputGradient[0, 0], 1e-6);
        Assert.AreEqual(1 * (1 - Math.Pow(Math.Tanh(1), 2)), inputGradient[0, 1], 1e-6);
        Assert.AreEqual(1 * (1 - Math.Pow(Math.Tanh(-1), 2)), inputGradient[1, 0], 1e-6);
        Assert.AreEqual(1 * (1 - Math.Pow(Math.Tanh(2), 2)), inputGradient[1, 1], 1e-6);
    }
}
