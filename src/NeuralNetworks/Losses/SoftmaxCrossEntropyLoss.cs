// Neural Networks in C♯
// File name: SoftmaxCrossEntropyLoss2.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core;

using static NeuralNetworks.Core.Arrays.OperationOps;

namespace NeuralNetworks.Losses;

/// <summary>
/// Categorical Cross-Entropy Loss combined with Softmax activation function.
/// </summary>
/// <param name="eps"></param>
public class SoftmaxCrossEntropyLoss(float eps = 1e-7f) : Loss2D
{
    private float[,]? _softmaxPrediction;

    protected override float CalculateLoss()
    {
        // Calculate the probabilities for the whole batch.
        _softmaxPrediction = Prediction.Softmax();

        //// Clip the probabilities to avoid log(0).
        //float[,] clippedSoftmax = _softmaxPrediction.Clip(eps, 1 - eps);

        //return -clippedSoftmax.Log().MultiplyElementwise(Target).Mean();
        return CrossEntropyLoss(_softmaxPrediction, Target, eps);
    }

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_softmaxPrediction != null, "_softmaxPrediction should not be null here.");

        int batchSize = Prediction.GetLength(0);
        return _softmaxPrediction.Subtract(Target).Divide(batchSize);
    }

    public override string ToString() => $"SoftmaxCrossEntropyLoss (eps={eps})";
}
