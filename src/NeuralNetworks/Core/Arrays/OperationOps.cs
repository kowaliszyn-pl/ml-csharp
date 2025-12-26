// Neural Networks in C♯
// File name: OperationOps.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

namespace NeuralNetworks.Core.Arrays;

public static class OperationOps
{
    public static float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f)
    {
        /* // Clip the probabilities to avoid log(0).
           float[,] clippedSoftmax = predicted.Clip(eps, 1 - eps);
           return -clippedSoftmax.Log().MultiplyElementwise(target).Mean();
        */

        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        float loss = 0f;
        int batchSize = predicted.GetLength(0);
        int numClasses = predicted.GetLength(1);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < numClasses; j++)
            {
                float p = Math.Clamp(predicted[i, j], eps, 1 - eps);
                loss += target[i, j] * MathF.Log(p);
            }
        }
        return -loss / (batchSize * numClasses);
    }

    public static float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target)
    {
        /* int batchSize = Prediction.GetLength(0);
            return _softmaxPrediction.Subtract(Target).Divide(batchSize);
        */

        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        int batchSize = predicted.GetLength(0);
        int numClasses = predicted.GetLength(1);
        float[,] gradient = new float[batchSize, numClasses];
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < numClasses; j++)
            {
                gradient[i, j] = (predicted[i, j] - target[i, j]) / batchSize;
            }
        }
        
        return gradient;
    }
}
