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

    public override string ToString()
        => "NullRegularizer";
}
