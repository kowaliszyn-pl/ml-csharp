// Neural Networks in C♯
// File name: NullRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

/// <summary>
/// Represents a regularizer that applies no regularization.
/// </summary>
/// <remarks>Use <see cref="Instance"/> to access the shared singleton instance.</remarks>
public sealed class NullRegularizer : BaseRegularizer
{
    /// <summary>
    /// Gets the singleton instance of <see cref="NullRegularizer"/>.
    /// </summary>
    public static NullRegularizer Instance { get; } = new NullRegularizer();

    private NullRegularizer(): base(false)
    {
    }

    public override string ToString()
        => "NullRegularizer";
}
