// Neural Networks in C♯
// File name: WeightMultiplyTests.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Regularization;

namespace NeuralNetworks.Operations.Parameterized.Tests;

[TestClass]
public class WeightMultiplyTests
{
    [TestMethod]
    public void Backward_WithoutRegularizer_LeavesParamGradientUnchanged()
    {
        // Arrange
        float[,] weights = { { 1f, -2f }, { 3f, 4f } };
        WeightMultiply weightMultiply = new(weights);
        float[,] input = { { 1f, 1f } };
        _ = weightMultiply.Forward(input, false);
        float[,] outputGradient = { { 1f, 1f } };

        // Act
        _ = weightMultiply.Backward(outputGradient);

        // Assert
        float[,] expectedParamGradient = WeightMultiplyParamGradientReference(input, outputGradient);
        AssertEqual2D(expectedParamGradient, weightMultiply.ParamGradient);
    }

    [TestMethod]
    public void Backward_WithRegularizer_AddsRegularizationToParamGradient()
    {
        // Arrange
        float[,] weights = { { 1f, -2f }, { 3f, 4f } };
        Regularizer regularizer = new(l1: 0.1f, l2: 0.2f);
        WeightMultiply weightMultiply = new(weights, regularizer);
        float[,] input = { { 1f, 1f } };
        _ = weightMultiply.Forward(input, false);
        float[,] outputGradient = { { 1f, 1f } };

        // Act
        _ = weightMultiply.Backward(outputGradient);

        // Assert
        float[,] baseParamGradient = WeightMultiplyParamGradientReference(input, outputGradient);
        float[,] expectedParamGradient = new float[2, 2];
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                float value = weights[i, j];
                expectedParamGradient[i, j] = baseParamGradient[i, j]
                    + regularizer.L1 * MathF.Sign(value)
                    + 2f * regularizer.L2 * value;
            }
        }
        AssertEqual2D(expectedParamGradient, weightMultiply.ParamGradient);
    }

    [TestMethod]
    public void GetRegularizationLoss_WithoutRegularizer_ReturnsZero()
    {
        // Arrange
        float[,] weights = { { 1f, -2f }, { 3f, 4f } };
        WeightMultiply weightMultiply = new(weights);

        // Act
        float loss = weightMultiply.GetRegularizationLoss();

        // Assert
        Assert.AreEqual(0f, loss);
    }

    [TestMethod]
    public void GetRegularizationLoss_WithRegularizerIncludeInLossFalse_ReturnsZero()
    {
        // Arrange
        float[,] weights = { { 1f, -2f }, { 3f, 4f } };
        Regularizer regularizer = new(l1: 0.1f, l2: 0.1f, reportPenaltyInLoss: false);
        WeightMultiply weightMultiply = new(weights, regularizer);

        // Act
        float loss = weightMultiply.GetRegularizationLoss();

        // Assert
        Assert.AreEqual(0f, loss);
    }

    [TestMethod]
    public void GetRegularizationLoss_WithRegularizerIncludeInLossTrue_ReturnsPenalty()
    {
        // Arrange
        float[,] weights = { { 1f, -2f }, { 3f, 4f } };
        Regularizer regularizer = new(l1: 0.1f, l2: 0.1f, reportPenaltyInLoss: true);
        WeightMultiply weightMultiply = new(weights, regularizer);

        // Act
        float loss = weightMultiply.GetRegularizationLoss();

        // Assert
        float expected = regularizer.CalculatePenalty(weights);
        Assert.AreEqual(expected, loss, 1e-6);
    }

    private static float[,] WeightMultiplyParamGradientReference(float[,] input, float[,] outputGradient)
    {
        // input^T * outputGradient
        int inputCols = input.GetLength(1);
        int outputCols = outputGradient.GetLength(1);
        int rows = input.GetLength(0);

        float[,] result = new float[inputCols, outputCols];
        for (int i = 0; i < inputCols; i++)
        {
            for (int j = 0; j < outputCols; j++)
            {
                float sum = 0f;
                for (int k = 0; k < rows; k++)
                {
                    sum += input[k, i] * outputGradient[k, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }

    private static void AssertEqual2D(float[,] expected, float[,] actual)
    {
        Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
        Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
        for (int i = 0; i < expected.GetLength(0); i++)
        {
            for (int j = 0; j < expected.GetLength(1); j++)
            {
                Assert.AreEqual(expected[i, j], actual[i, j], 1e-5, $"Mismatch at [{i},{j}]");
            }
        }
    }
}
