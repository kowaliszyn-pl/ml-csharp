// Neural Networks in C♯
// File name: ElasticNetRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

/// <summary>
/// Combines L1 and L2 regularization into a single elastic net regularizer.
/// </summary>
/// <remarks>Uses separate L1 and L2 regularizers (<see cref="L1Regularizer"/> and <see cref="L2Regularizer"/>) and applies both penalties during training.</remarks>
/// <param name="l1">L1 regularization coefficient.</param>
/// <param name="l2">L2 regularization coefficient.</param>
/// <param name="reportPenaltyInLoss">Indicates whether the regularization penalty is included in the reported loss value.</param>
public class ElasticNetRegularizer(float l1, float l2, bool reportPenaltyInLoss = false) 
    : CompositeRegularizer(new L1Regularizer(l1, reportPenaltyInLoss), new L2Regularizer(l2, reportPenaltyInLoss))
{
    public float L1 => l1;

    public float L2 => l2;

    public override string ToString()
        => $"ElasticNetRegularizer(l1={L1}, l2={L2}, reportPenaltyInLoss={ReportPenaltyInLoss})";
}
