// Neural Networks in C♯
// File name: NullRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

public class NullRegularizer : IRegularizer
{
    public static NullRegularizer Instance { get; } = new NullRegularizer();

    public bool ReportPenaltyInLoss => false;

    public void ApplyGradient<T>(T param, T paramGradient) where T : notnull
    {
    }

    public float ReportPenalty<T>(T param) where T : notnull
        => 0f;

    public void UpdateParams<T>(T param) where T : notnull
    {
    }
}
