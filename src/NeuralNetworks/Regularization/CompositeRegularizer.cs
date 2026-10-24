// Neural Networks in C♯
// File name: CompositeRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Core;

namespace NeuralNetworks.Regularization;

/// <summary>
/// Combines multiple regularizers into a single regularization pipeline.
/// </summary>
/// <remarks>Applies regularizers in the order provided for gradient adjustment, penalty calculation, and
/// parameter updates. Penalty reporting is enabled when at least one contained regularizer reports a penalty in the
/// loss, and the reported penalty is the sum of all enabled regularizer penalties.</remarks>
public class CompositeRegularizer : IRegularizer
{
    private readonly BaseRegularizer[] _regularizers;

    private readonly bool _reportAnyPenaltyInLoss;

    /// <summary>
    /// Indicates whether penalty is reported as a loss.
    /// </summary>
    public bool ReportPenaltyInLoss => _reportAnyPenaltyInLoss;

    /// <param name="regularizers">
    /// The regularizers to combine. They are applied in the given order when computing gradients and penalties.
    /// </param>
    public CompositeRegularizer(params BaseRegularizer[] regularizers)
    {
        _regularizers = regularizers;
        _reportAnyPenaltyInLoss = Array.Exists(_regularizers, r => r.ReportPenaltyInLoss);
    }

    /// <summary>
    /// Applies a gradient update using the specified parameter values and corresponding gradient values.
    /// </summary>
    /// <typeparam name="T">The type of the parameter and gradient. Must be a non-null type.</typeparam>
    /// <param name="param">The parameter value to adjust.</param>
    /// <param name="paramGradient">The gradient of the parameter to adjust.</param>
    /// <remarks>
    /// It calls the <see cref="BaseRegularizer.ApplyGradientCore"/> method of each contained regularizer in the order they were provided, allowing for sequential gradient adjustments.
    /// </remarks>
    public void ApplyGradient<T>(T param, T paramGradient) where T : notnull
    {
        ReadOnlySpan<float> paramSpan = GenericArrayRoutines.ConvertToReadOnlySpan(param);
        Span<float> paramGradientSpan = GenericArrayRoutines.ConvertToSpan(paramGradient);

        Debug.Assert(paramSpan.Length == paramGradientSpan.Length, "Parameter and its gradient must have the same length.");

        foreach (BaseRegularizer regularizer in _regularizers)
        {
            regularizer.ApplyGradientCore(paramSpan, paramGradientSpan);
        }
    }

    /// <summary>
    /// Calculates the total regularization penalty to include in the loss for the provided parameter values.
    /// </summary>
    /// <remarks>Returns <c>0</c> when penalty reporting in loss is disabled.</remarks>
    /// <typeparam name="T">The parameter container type that can be converted to a read-only span of single-precision values.</typeparam>
    /// <param name="param">The parameter values used to evaluate regularization penalties.</param>
    /// <returns>The sum of penalties from regularizers configured to report penalty in loss; otherwise <c>0</c>.</returns>
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

    /// <summary>
    /// Updates all configured regularizers with the provided parameter values.
    /// </summary>
    /// <remarks>Converts the input to a <see cref="System.Span{System.Single}"/> and forwards it to each
    /// regularizer in sequence.</remarks>
    /// <typeparam name="T">A non-null parameter container type supported by the parameter-to-span conversion routine.</typeparam>
    /// <param name="param">The parameter data to convert and pass to each regularizer.</param>
    public void UpdateParams<T>(T param) where T : notnull
    {
        Span<float> paramSpan = GenericArrayRoutines.ConvertToSpan(param);
        foreach (BaseRegularizer regularizer in _regularizers)
        {
            regularizer.UpdateParamsCore(paramSpan);
        }
    }

    public override string ToString()
    {
        string regularizersStr = string.Join(", ", _regularizers.Select(r => r.ToString()));
        return $"CompositeRegularizer([{regularizersStr}])";
    }
}
