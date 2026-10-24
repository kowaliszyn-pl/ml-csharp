// Neural Networks in C♯
// File name: IRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

public interface IRegularizer
{
    public void ApplyGradient<T>(T param, T paramGradient) where T : notnull;

    public float ReportPenalty<T>(T param) where T : notnull;

    /// <summary>
    /// Applied to the parameters after the optimizer has updated them (e.g., for norm-constraint projections).
    /// Does nothing by default.
    /// </summary>
    public void UpdateParams<T>(T param) where T : notnull;

    public bool ReportPenaltyInLoss { get; }
}
