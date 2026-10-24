// Neural Networks in C♯
// File name: NullRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

public sealed class NullRegularizer : BaseRegularizer
{
    public static NullRegularizer Instance { get; } = new NullRegularizer();

    private NullRegularizer(): base(false)
    {
    }

    protected internal override float CalculatePenaltyCore(ReadOnlySpan<float> paramSpan)
        => 0f;
}
