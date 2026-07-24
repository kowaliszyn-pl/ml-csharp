// Neural Networks in C♯
// File name: CrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

/// <summary>
/// Categorical Cross-Entropy Loss without Softmax activation function.
/// </summary>
public class CrossEntropyLoss() : Loss<float[,]>
{
    private float[,]? _softmaxOutput;

    protected override float CalculateLoss()
        => CrossEntropyLoss(Prediction, Target, out _softmaxOutput);

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_softmaxOutput != null, "_softmaxOutput should not be null here.");

        return CrossEntropyLossGradient(_softmaxOutput, Target);
    }

    public override string ToString() => $"CrossEntropyLoss";   
}
