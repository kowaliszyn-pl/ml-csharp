// Neural Networks in C♯
// File name: OperationOps.cs
// www.kowaliszyn.pl, 2025

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NeuralNetworks.Core.Arrays;

public static class OperationOps
{
    public static float CrossEntropyLoss(float[,] predicted, float[] target, float eps = 1e-7f)
    {
        /* // Clip the probabilities to avoid log(0).
           float[,] clippedSoftmax = predicted.Clip(eps, 1 - eps);
           return -clippedSoftmax.Log().MultiplyElementwise(target).Mean();
        */

        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        float loss = 0f;
        for (int i = 0; i < predicted.Length; i++)
        {
            float p = Math.Clamp(predicted[i], eps, 1 - eps);
            loss += -target[i] * (float)Math.Log(p);
        }
        return loss / predicted.Length;
    }
}
