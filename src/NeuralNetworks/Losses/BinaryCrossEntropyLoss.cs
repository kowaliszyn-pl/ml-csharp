// Neural Networks in C♯
// File name: BinaryCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

public class BinaryCrossEntropyLoss : Loss<float[,]>
{
    protected override float CalculateLoss()
        => BinaryCrossEntropyLoss(Prediction, Target);

    protected override float[,] CalculateLossGradient()
        => BinaryCrossEntropyLossGradient(Prediction, Target);

    public override string ToString() => $"BinaryCrossEntropyLoss (eps={eps})";
}
