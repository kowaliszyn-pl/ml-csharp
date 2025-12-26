// Neural Networks in C♯
// File name: OperationOps.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Arrays;

public static class OperationOps
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        // Clip the probabilities to avoid log(0).
        float[,] clippedSoftmax = predicted.Clip(eps, 1 - eps);
        return -clippedSoftmax.Log().MultiplyElementwise(target).Mean();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        int batchSize = predicted.GetLength(0);
        return predicted.Subtract(target).Divide(batchSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] WeightMultiplyCalcOutput(float[,] input, float[,] weights)
        => input.MultiplyDot(weights);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] WeightMultiplyCalcInputGradient(float[,] outputGradient, float[,] weights) 
        => outputGradient.MultiplyDot(weights.Transpose());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] WeightMultiplyCalcParamGradient(float[,] input, float[,] outputGradient) 
        => input.Transpose().MultiplyDot(outputGradient);
}
