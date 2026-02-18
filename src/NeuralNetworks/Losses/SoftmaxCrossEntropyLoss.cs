// Neural Networks in C♯
// File name: SoftmaxCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Categorical Cross-Entropy Loss combined with Softmax activation function.
/// </summary>
/// <param name="eps"></param>
public class SoftmaxCrossEntropyLoss(float eps = 1e-7f) : Loss<float[,]>
{
    private float[,]? _softmaxPrediction;

    protected override float CalculateLoss()
    {
        // Calculate the probabilities for the whole batch.
        _softmaxPrediction = Prediction.Softmax();
        return CrossEntropyLoss(_softmaxPrediction, Target, eps);
    }

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_softmaxPrediction != null, "_softmaxPrediction should not be null here.");

        return CrossEntropyLossGradient(_softmaxPrediction, Target);
    }

    public override string ToString() => $"SoftmaxCrossEntropyLoss (eps={eps})";
}
