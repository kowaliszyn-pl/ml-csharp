// Neural Networks in C♯
// File name: CompositeRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

namespace NeuralNetworks.Regularization;

public class CompositeRegularizer : IRegularizer
{
    private readonly BaseRegularizer[] _regularizers;

    private readonly bool _reportAnyPenaltyInLoss;

    public bool ReportPenaltyInLoss => _reportAnyPenaltyInLoss;

    /// <param name="regularizers">
    /// The regularizers to combine. They are applied in the given order when computing gradients and penalties.
    /// </param>
    public CompositeRegularizer(params BaseRegularizer[] regularizers)
    {
        _regularizers = regularizers;
        _reportAnyPenaltyInLoss = Array.Exists(_regularizers, r => r.ReportPenaltyInLoss);
    }

    public void ApplyGradient<T>(T param, T paramGradient) where T : notnull
    {
        ReadOnlySpan<float> paramSpan = GenericArrayRoutines.ConvertToReadOnlySpan(param);
        Span<float> paramGradientSpan = GenericArrayRoutines.ConvertToSpan(paramGradient);
        foreach (BaseRegularizer regularizer in _regularizers)
        {
            regularizer.ApplyGradientCore(paramSpan, paramGradientSpan);
        }
    }

    public float ReportPenalty<T>(T param) where T : notnull
    {
        if (_reportAnyPenaltyInLoss)
        {
            ReadOnlySpan<float> paramSpan = GenericArrayRoutines.ConvertToReadOnlySpan(param);
            float totalPenalty = 0f;
            foreach (BaseRegularizer regularizer in _regularizers)
            {
                if (regularizer.ReportPenaltyInLoss)
                {
                    totalPenalty += regularizer.CalculatePenaltyCore(paramSpan);
                }
            }
            return totalPenalty;
        }
        return 0f;
    }

    public void UpdateParams<T>(T param) where T : notnull
    {
        Span<float> paramSpan = GenericArrayRoutines.ConvertToSpan(param);
        foreach (BaseRegularizer regularizer in _regularizers)
        {
            regularizer.UpdateParamsCore(paramSpan);
        }
    }
}
