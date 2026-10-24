// Neural Networks in C♯
// File name: CompositeRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

public class CompositeRegularizer : BaseRegularizer
{
    private readonly BaseRegularizer[] _regularizers;

    /// <param name="regularizers">
    /// The regularizers to combine. They are applied in the given order when computing gradients and penalties.
    /// </param>
    public CompositeRegularizer(params BaseRegularizer[] regularizers) : base(AnyReportPenaltyInLoss(regularizers))
    {
        _regularizers = regularizers;
    }

    private static bool AnyReportPenaltyInLoss(BaseRegularizer[] regularizers) 
        => Array.Exists(regularizers, r => r.ReportPenaltyInLoss);

    protected internal override void ApplyGradientCore(Span<float> paramSpan, Span<float> paramGradientSpan)
    {
        foreach (BaseRegularizer regularizer in _regularizers)
        {
            regularizer.ApplyGradientCore(paramSpan, paramGradientSpan);
        }
    }

    protected internal override float CalculatePenalty(ReadOnlySpan<float> paramSpan)
    {
        float total = 0f;
        foreach (BaseRegularizer regularizer in _regularizers)
        {
            total += regularizer.CalculatePenalty(paramSpan);
        }
        return total;
    }
}
