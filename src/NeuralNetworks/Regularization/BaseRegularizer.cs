// Neural Networks in C♯
// File name: BaseRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;

namespace NeuralNetworks.Regularization;

public abstract class BaseRegularizer(bool reportPenaltyInLoss) : IRegularizer
{
    public bool ReportPenaltyInLoss => reportPenaltyInLoss;

    public void ApplyGradient<T>(T param, T paramGradient) where T : notnull
    {
        Span<float> paramSpan = GenericArrayRoutines.ConvertToSpan(param);
        Span<float> paramGradientSpan = GenericArrayRoutines.ConvertToSpan(paramGradient);
        ApplyGradient(paramSpan, paramGradientSpan);
    }

    public float ReportPenalty<T>(T param) where T : notnull
    {
        if (reportPenaltyInLoss)
        {
            Span<float> paramSpan = GenericArrayRoutines.ConvertToSpan(param);
            return CalculatePenalty(paramSpan);
        }
        return 0f;
    }

    public void UpdateParams<T>(T param) where T : notnull
    {
        Span<float> paramSpan = GenericArrayRoutines.ConvertToSpan(param);
        UpdateParams(paramSpan);
    }

    protected abstract void ApplyGradient(Span<float> paramSpan, Span<float> paramGradientSpan);
    protected abstract float CalculatePenalty(ReadOnlySpan<float> paramSpan);
    protected virtual void UpdateParams(Span<float> paramSpan) { }
}
