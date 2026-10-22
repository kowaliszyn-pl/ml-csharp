// Neural Networks in C♯
// File name: RegularizerTests.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Regularization;

namespace NeuralNetworks.Regularization.Tests;

[TestClass]
public class RegularizerTests
{
    [TestMethod]
    public void ApplyGradient_L1Only_AddsSignContribution()
    {
        // Arrange
        Regularizer regularizer = new(l1: 0.1f, l2: 0f);
        float[] param = [2f, -3f, 0f];
        float[] paramGradient = [1f, 1f, 1f];

        // Act
        regularizer.ApplyGradient(param, paramGradient);

        // Assert
        Assert.AreEqual(1f + 0.1f * 1f, paramGradient[0], 1e-6);
        Assert.AreEqual(1f + 0.1f * -1f, paramGradient[1], 1e-6);
        Assert.AreEqual(1f + 0.1f * 0f, paramGradient[2], 1e-6);
    }

    [TestMethod]
    public void ApplyGradient_L2Only_AddsScaledValueContribution()
    {
        // Arrange
        Regularizer regularizer = new(l1: 0f, l2: 0.1f);
        float[] param = [2f, -3f];
        float[] paramGradient = [1f, 1f];

        // Act
        regularizer.ApplyGradient(param, paramGradient);

        // Assert
        Assert.AreEqual(1f + 2f * 0.1f * 2f, paramGradient[0], 1e-6);
        Assert.AreEqual(1f + 2f * 0.1f * -3f, paramGradient[1], 1e-6);
    }

    [TestMethod]
    public void ApplyGradient_CombinedL1AndL2_AddsBothContributions()
    {
        // Arrange
        Regularizer regularizer = new(l1: 0.1f, l2: 0.2f);
        float[] param = [2f];
        float[] paramGradient = [1f];

        // Act
        regularizer.ApplyGradient(param, paramGradient);

        // Assert
        float expected = 1f + 0.1f * 1f + 2f * 0.2f * 2f;
        Assert.AreEqual(expected, paramGradient[0], 1e-6);
    }

    [TestMethod]
    public void ApplyGradient_ZeroCoefficients_NoOp()
    {
        // Arrange
        Regularizer regularizer = new(l1: 0f, l2: 0f);
        float[] param = [2f, -3f];
        float[] paramGradient = [1f, 1f];

        // Act
        regularizer.ApplyGradient(param, paramGradient);

        // Assert
        Assert.AreEqual(1f, paramGradient[0], 1e-6);
        Assert.AreEqual(1f, paramGradient[1], 1e-6);
    }

    [TestMethod]
    public void CalculatePenalty_L1Only_ReturnsSumOfAbsoluteValues()
    {
        // Arrange
        Regularizer regularizer = new(l1: 0.5f, l2: 0f);
        float[] param = [2f, -3f];

        // Act
        float penalty = regularizer.CalculatePenalty(param);

        // Assert
        Assert.AreEqual(0.5f * (2f + 3f), penalty, 1e-6);
    }

    [TestMethod]
    public void CalculatePenalty_L2Only_ReturnsSumOfSquares()
    {
        // Arrange
        Regularizer regularizer = new(l1: 0f, l2: 0.5f);
        float[] param = [2f, -3f];

        // Act
        float penalty = regularizer.CalculatePenalty(param);

        // Assert
        Assert.AreEqual(0.5f * (4f + 9f), penalty, 1e-6);
    }

    [TestMethod]
    public void CalculatePenalty_Combined_ReturnsSumOfBoth()
    {
        // Arrange
        Regularizer regularizer = new(l1: 0.1f, l2: 0.2f);
        float[] param = [2f, -3f];

        // Act
        float penalty = regularizer.CalculatePenalty(param);

        // Assert
        float expected = 0.1f * (2f + 3f) + 0.2f * (4f + 9f);
        Assert.AreEqual(expected, penalty, 1e-6);
    }

    [TestMethod]
    public void IncludeInLossValue_DefaultsToFalse()
    {
        // Arrange
        Regularizer regularizer = new(l1: 0.1f, l2: 0.1f);

        // Act & Assert
        Assert.IsFalse(regularizer.IncludeInLossValue);
    }

    [TestMethod]
    public void IncludeInLossValue_CanBeSetToTrue()
    {
        // Arrange
        Regularizer regularizer = new(l1: 0.1f, l2: 0.1f, includeInLossValue: true);

        // Act & Assert
        Assert.IsTrue(regularizer.IncludeInLossValue);
    }
}
