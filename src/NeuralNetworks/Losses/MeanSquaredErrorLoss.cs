// Neural Networks in C♯
// File name: MeanSquaredErrorLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

public class MeanSquaredErrorLoss : Loss<float[,]>
{
    private float[,]? _errors;

    protected override float CalculateLoss()
        => MeanSquaredErrorLoss(Prediction, Target, out _errors);

    protected override float[,] CalculateLossGradient()
    {
        Debug.Assert(_errors != null, "_errors should not be null here.");

        return MeanSquaredErrorLossGradient(_errors);
    }

    public override string ToString() => "MeanSquaredError";
}