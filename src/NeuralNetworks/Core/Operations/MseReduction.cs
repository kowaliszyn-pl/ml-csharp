// Neural Networks in C♯
// File name: MseReduction.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Core.Operations;

public enum MseReduction
{
    /// <summary>
    /// Divides total squared error by batch size.
    /// </summary>
    BatchMean,

    /// <summary>
    /// Divides total squared error by the total element count (batch size * number of features).
    /// </summary>
    ElementMean
}