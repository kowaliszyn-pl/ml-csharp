// Neural Networks in C♯
// File name: MeanSquaredErrorLoss4D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Core;

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

        //int batchSize = Prediction.GetLength(0);
        //_errors = Prediction.Subtract(Target);
        //// The quadratic function has the property that values further from the minimum have a steeper gradient.
        //return _errors.Power(2).Sum() / batchSize;
    }

    protected override float[,,,] CalculateLossGradient()
    {
        Debug.Assert(_errors != null, "_errors should not be null here.");

        return MeanSquaredErrorLossGradient(Prediction, _errors);

        //int batchSize = Prediction.GetLength(0);
        //return _errors.Multiply(2f / batchSize);
    }

    public override string ToString() => "MeanSquaredError4D";
}
