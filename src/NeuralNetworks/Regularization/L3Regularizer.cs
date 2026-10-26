// Neural Networks in C♯
// File name: L3Regularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

/// <summary>
/// Applies L3 regularization by adding a penalty proportional to the cubed magnitude of model parameters.
/// </summary>
/// <remarks>Uses the L3 penalty form l3 * Σ(|w|^3), contributes 3 * l3 * w * |w| to each parameter gradient,
/// and can optionally include the penalty term in the reported loss via the base regularizer configuration.</remarks>
public class L3Regularizer : BaseRegularizer
{
    private readonly float _l3;
    private readonly float _l3TimesThree;

    public L3Regularizer(float l3, bool reportPenaltyInLoss = false) : base(reportPenaltyInLoss)
    {
        if (l3 <= 0f)
            throw new ArgumentOutOfRangeException(nameof(l3), "L3 regularization parameter must be positive.");

        _l3 = l3;
        _l3TimesThree = 3f * l3;
    }

    protected internal override void ApplyGradientCore(ReadOnlySpan<float> paramSpan, Span<float> paramGradientSpan)
    {
        for (int i = 0; i < paramSpan.Length; i++)
        {
            float value = paramSpan[i];
            float absValue = MathF.Abs(value);
            paramGradientSpan[i] += _l3TimesThree * value * absValue;
        }
    }

    protected internal override float CalculatePenaltyCore(ReadOnlySpan<float> paramSpan)
    {
        float sum = 0f;
        foreach (float paramValue in paramSpan)
        {
            float absValue = MathF.Abs(paramValue);
            sum += absValue * absValue * absValue;
        }
        return _l3 * sum;
    }

    public override string ToString() => $"L3Regularizer(l3={_l3}, reportPenaltyInLoss={ReportPenaltyInLoss})";
}
