// Neural Networks in C♯
// File name: SoftmaxCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;

namespace NeuralNetworks.Losses;

public class SoftmaxCrossEntropyLoss(float eps = 1e-7f) : Loss2D
{
    protected override float CalculateLoss()
    {
        // Calculate the probabilities for the whole batch.
        float[,] softmaxPrediction = Prediction.Softmax();

        // Clip the probabilities to avoid log(0).
        softmaxPrediction.ClipInPlace(eps, 1 - eps);

        float[,] negativeTarget = Target.Multiply(-1f);
        float[,] softmaxCrossEntropyLoss = negativeTarget.MultiplyElementwise(softmaxPrediction.Log())
            .Subtract(
                negativeTarget.Add(1f).MultiplyElementwise(softmaxPrediction.Multiply(-1f).Add(1f).Log())
            );
        int batchSize = Prediction.GetLength(0);
        return softmaxCrossEntropyLoss.Sum() / batchSize;
    }

    protected override float[,] CalculateLossGradient()
    {
        float[,] softmaxPrediction = Prediction.Softmax();
        int batchSize = Prediction.GetLength(0);
        return softmaxPrediction.Subtract(Target).Divide(batchSize);
    }

    override public string ToString() => $"SoftmaxCrossEntropyLoss (eps={eps})";
}
