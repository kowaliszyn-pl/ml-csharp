// Neural Networks in C♯
// File name: L1Regularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

public class L1Regularizer : BaseRegularizer
{

    private readonly float _l1;

    public L1Regularizer(float l1, bool reportPenaltyInLoss = false) : base(reportPenaltyInLoss)
    {
        if (l1 <= 0f)
            throw new ArgumentOutOfRangeException(nameof(l1), "L1 regularization parameter must be positive.");

        _l1 = l1;
    }

    protected internal override void ApplyGradientCore(Span<float> paramSpan, Span<float> paramGradientSpan)
    {
        for (int i = 0; i < paramSpan.Length; i++)
        {
            paramGradientSpan[i] += _l1 * MathF.Sign(paramSpan[i]);
        }
    }

    protected internal override float CalculatePenalty(ReadOnlySpan<float> paramSpan)
    {
        float sum = 0f;
        foreach (float paramValue in paramSpan)
        {
            sum += MathF.Abs(paramValue);
        }
        return _l1 * sum;
    }
}
