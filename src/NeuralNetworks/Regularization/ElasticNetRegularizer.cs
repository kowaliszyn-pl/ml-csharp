// Neural Networks in C♯
// File name: ElasticNetRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

public class ElasticNetRegularizer(float l1, float l2, bool reportPenaltyInLoss = false) 
    : CompositeRegularizer(new L1Regularizer(l1, reportPenaltyInLoss), new L2Regularizer(l2, reportPenaltyInLoss))
{
    public float L1 => l1;

    public float L2 => l2;
}
