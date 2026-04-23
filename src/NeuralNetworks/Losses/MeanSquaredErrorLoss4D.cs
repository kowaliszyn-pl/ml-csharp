// Neural Networks in C♯
// File name: MeanSquaredErrorLoss4D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

public class MeanSquaredErrorLoss4D : Loss<float[,,,]>
{
    private float[,,,]? _errors;

    protected override float CalculateLoss()
    {
        float loss = MeanSquaredErrorLoss(Prediction, Target, out float[,,,] errors);
        _errors = errors;
        return loss;
    }

    protected override float[,,,] CalculateLossGradient()
    {
        Debug.Assert(_errors != null, "_errors should not be null here.");

        return MeanSquaredErrorLossGradient(Prediction, _errors);
    }

    public override string ToString() => "MeanSquaredError4D";
}
