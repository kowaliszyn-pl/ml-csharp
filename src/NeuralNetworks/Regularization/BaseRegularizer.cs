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
        ReadOnlySpan<float> paramSpan = GenericArrayRoutines.ConvertToReadOnlySpan(param);
        Span<float> paramGradientSpan = GenericArrayRoutines.ConvertToSpan(paramGradient);
        ApplyGradientCore(paramSpan, paramGradientSpan);
    }

    public float ReportPenalty<T>(T param) where T : notnull
    {
        if (reportPenaltyInLoss)
        {
            ReadOnlySpan<float> paramSpan = GenericArrayRoutines.ConvertToReadOnlySpan(param);
            return CalculatePenaltyCore(paramSpan);
        }
        return 0f;
    }

    public void UpdateParams<T>(T param) where T : notnull
    {
        Span<float> paramSpan = GenericArrayRoutines.ConvertToSpan(param);
        UpdateParamsCore(paramSpan);
    }

    protected internal virtual void ApplyGradientCore(ReadOnlySpan<float> paramSpan, Span<float> paramGradientSpan)
    {
    }

    protected internal abstract float CalculatePenaltyCore(ReadOnlySpan<float> paramSpan);

    protected internal virtual void UpdateParamsCore(Span<float> paramSpan)
    {
    }
}
