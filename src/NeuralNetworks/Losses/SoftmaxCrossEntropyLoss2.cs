// Neural Networks in C♯
// File name: SoftmaxCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;

namespace NeuralNetworks.Losses;

public class SoftmaxCrossEntropyLoss2(float eps = 1e-7f) : Loss2D
{
    protected override float CalculateLoss()
    {
        // Calculate the probabilities for the whole batch.
        float[,] softmaxPrediction = Prediction.Softmax();
        
        // Clip the probabilities to avoid log(0).
        softmaxPrediction.ClipInPlace(eps, 1 - eps);

        int batchSize = Prediction.GetLength(0);

        // return -(softmaxPrediction.Log().MultiplyElementwise(Target).Sum() / batchSize);
        return -softmaxPrediction.Log().MultiplyElementwise(Target).Mean();

        /*
        float loss = 0f;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < Target.GetLength(1); j++)
            {
                if (Target[i, j] == 1f)
                {
                    loss += -MathF.Log(softmaxPrediction[i, j]);
                }
            }
        }
        return loss / batchSize;*/
    }

    protected override float[,] CalculateLossGradient()
    {
        float[,] softmaxPrediction = Prediction.Softmax();
        int batchSize = Prediction.GetLength(0);

        float[,] lossGradient = new float[batchSize, Target.GetLength(1)];
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < Target.GetLength(1); j++)
            {
                lossGradient[i, j] = softmaxPrediction[i, j] - Target[i, j];
            }
        }
        return lossGradient.Divide(batchSize);
    }

    override public string ToString() => $"SoftmaxCrossEntropyLoss (eps={eps})";
}
