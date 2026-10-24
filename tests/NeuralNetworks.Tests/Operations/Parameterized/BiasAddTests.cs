// Neural Networks in C♯
// File name: BiasAddTests.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Regularization;

namespace NeuralNetworks.Operations.Parameterized.Tests;

[TestClass]
public class BiasAddTests
{
    [TestMethod]
    public void Backward_WithoutRegularizer_LeavesParamGradientUnchanged()
    {
        // Arrange
        float[] bias = [1f, -2f];
        BiasAdd biasAdd = new(bias);
        float[,] input = { { 1f, 1f } };
        _ = biasAdd.Forward(input, false);
        float[,] outputGradient = { { 2f, 3f }, { 4f, 5f } };

        // Act
        _ = biasAdd.Backward(outputGradient);

        // Assert
        float[] expectedParamGradient = BiasAddParamGradientReference(outputGradient);
        AssertEqual1D(expectedParamGradient, biasAdd.ParamGradient);
    }

    [TestMethod]
    public void Backward_WithRegularizer_AddsRegularizationToParamGradient()
    {
        // Arrange
        float[] bias = [1f, -2f];
        ElasticNetRegularizer regularizer = new(l1: 0.1f, l2: 0.2f);
        BiasAdd biasAdd = new(bias, regularizer);
        float[,] input = { { 1f, 1f } };
        _ = biasAdd.Forward(input, false);
        float[,] outputGradient = { { 2f, 3f }, { 4f, 5f } };

        // Act
        _ = biasAdd.Backward(outputGradient);

        // Assert
        float[] baseParamGradient = BiasAddParamGradientReference(outputGradient);
        float[] expectedParamGradient = new float[2];
        for (int i = 0; i < 2; i++)
        {
            float value = bias[i];
            expectedParamGradient[i] = baseParamGradient[i]
                + regularizer.L1 * MathF.Sign(value)
                + 2f * regularizer.L2 * value;
        }
        AssertEqual1D(expectedParamGradient, biasAdd.ParamGradient);
    }

    [TestMethod]
    public void GetRegularizationLoss_WithoutRegularizer_ReturnsZero()
    {
        // Arrange
        float[] bias = [1f, -2f];
        BiasAdd biasAdd = new(bias);

        // Act
        float loss = biasAdd.GetRegularizationLoss();

        // Assert
        Assert.AreEqual(0f, loss);
    }

    [TestMethod]
    public void GetRegularizationLoss_WithRegularizerIncludeInLossTrue_ReturnsPenalty()
    {
        // Arrange
        float[] bias = [1f, -2f];
        ElasticNetRegularizer regularizer = new(l1: 0.1f, l2: 0.1f, reportPenaltyInLoss: true);
        BiasAdd biasAdd = new(bias, regularizer);

        // Act
        float loss = biasAdd.GetRegularizationLoss();

        // Assert
        float expected = regularizer.CalculatePenalty(bias);
        Assert.AreEqual(expected, loss, 1e-6);
    }

    private static float[] BiasAddParamGradientReference(float[,] outputGradient)
    {
        int rows = outputGradient.GetLength(0);
        int cols = outputGradient.GetLength(1);
        float[] result = new float[cols];
        for (int j = 0; j < cols; j++)
        {
            float sum = 0f;
            for (int i = 0; i < rows; i++)
            {
                sum += outputGradient[i, j];
            }
            result[j] = sum;
        }
        return result;
    }

    private static void AssertEqual1D(float[] expected, float[] actual)
    {
        Assert.HasCount(expected.Length, actual);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.AreEqual(expected[i], actual[i], 1e-5, $"Mismatch at [{i}]");
        }
    }
}
