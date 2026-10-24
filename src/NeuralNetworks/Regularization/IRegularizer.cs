// Neural Networks in C♯
// File name: IRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

/// <summary>
/// Defines regularization operations for applying gradient-based adjustments, reporting penalty terms, and updating
/// parameters after optimizer steps.
/// </summary>
/// <remarks>Implementations encapsulate regularization strategy behavior used during and after optimization
/// updates, including optional constraint projections.</remarks>
public interface IRegularizer
{
    /// <summary>
    /// Applies a gradient update using the specified parameter values and corresponding gradient values.
    /// </summary>
    /// <typeparam name="T">The type of the parameter and gradient. Must be a non-null type.</typeparam>
    /// <param name="param">The parameter value to adjust.</param>
    /// <param name="paramGradient">The gradient of the parameter to adjust.</param>
    public void ApplyGradient<T>(T param, T paramGradient) where T : notnull;

    /// <summary>
    /// Reports a penalty for the specified value.
    /// </summary>
    /// <typeparam name="T">The non-nullable type of value used to report the penalty.</typeparam>
    /// <param name="param">The value used to report the penalty.</param>
    /// <returns>The reported penalty value.</returns>
    public float ReportPenalty<T>(T param) where T : notnull;

    /// <summary>
    /// Applied to the parameters after the optimizer has updated them (e.g., for norm-constraint projections).
    /// </summary>
    /// <typeparam name="T">The type of the parameter to update. Must be a non-null type.</typeparam>
    /// <param name="param">The parameter value to update.</param>
    public void UpdateParams<T>(T param) where T : notnull;
}
