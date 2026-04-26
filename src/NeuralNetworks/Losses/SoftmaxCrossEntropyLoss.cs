// Neural Networks in C♯
// File name: SoftmaxCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Categorical Cross-Entropy Loss combined with Softmax activation function.
/// </summary>
/// <param name="eps"></param>
/// <remarks>
/// Usually used in MNIST-like classification problems, where the target is a one-hot encoded vector and the prediction is a probability distribution over classes.
/// </remarks>
public class SoftmaxCrossEntropyLoss(float eps = 1e-7f) : Loss<float[,]>
{
    private float[,]? _softmaxPrediction;

    protected override float CalculateLoss() 
        => SoftmaxCrossEntropyLoss(Prediction, Target, out _softmaxPrediction, eps);

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_softmaxPrediction != null, "_softmaxPrediction should not be null here.");

        return SoftmaxCrossEntropyLossGradient(_softmaxPrediction, Target);
    }

    public override string ToString() => $"SoftmaxCrossEntropyLoss (eps={eps})";
}
