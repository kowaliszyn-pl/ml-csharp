// Neural Networks in C♯
// File name: L2Regularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

public class L2Regularizer : BaseRegularizer
{
    private readonly float _l2;

    public L2Regularizer(float l2, bool reportPenaltyInLoss = true) : base(reportPenaltyInLoss)
    {
        if (l2 <= 0f)
            throw new ArgumentOutOfRangeException(nameof(l2), "L2 regularization parameter must be positive.");

        _l2 = l2;
    }

    protected override void ApplyGradient(Span<float> paramSpan, Span<float> paramGradientSpan)
    {
        for (int i = 0; i < paramSpan.Length; i++)
        {
            paramGradientSpan[i] += 2f * _l2 * paramSpan[i];
        }
    }

    protected override float CalculatePenalty(ReadOnlySpan<float> paramSpan)
    {
        float sum = 0f;
        foreach (float paramValue in paramSpan)
        {
            sum += paramValue * paramValue;
        }
        return _l2 * sum;
    }
}
