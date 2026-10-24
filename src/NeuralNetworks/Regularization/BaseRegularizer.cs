// Neural Networks in C♯
// File name: BaseRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Core;

namespace NeuralNetworks.Regularization;

/// <summary>
/// Provides a base implementation of <see cref="IRegularizer"/> for applying regularization updates and optionally
/// reporting a regularization penalty in the loss.
/// </summary>
/// <remarks>Derived types implement <see cref="CalculatePenaltyCore"/> and can override gradient and parameter update
/// hooks to define regularization behavior.
/// </remarks>
/// <param name="reportPenaltyInLoss">Indicates whether the regularization penalty is included when reporting loss during training.</param>
public abstract class BaseRegularizer(bool reportPenaltyInLoss) : IRegularizer
{
    /// <summary>
    /// Gets a value indicating whether a penalty is reported as a loss.
    /// </summary>
    /// <remarks>
    /// When it is <see langword="true"/>, the regularization penalty is intended to be included in the
    /// reported training loss; otherwise, regularization affects gradients without changing the reported loss
    /// value.
    /// </remarks>
    public bool ReportPenaltyInLoss => reportPenaltyInLoss;

    /// <summary>
    /// Applies a gradient update using the specified parameter values and corresponding gradient values.
    /// </summary>
    /// <remarks>Both inputs are converted to spans and must have the same length.</remarks>
    /// <typeparam name="T">The parameter container type that can be converted to spans of single-precision floating-point values.</typeparam>
    /// <param name="param">The parameter values to use for the update.</param>
    /// <param name="paramGradient">The gradient values corresponding to <paramref name="param"/>.</param>
    public void ApplyGradient<T>(T param, T paramGradient) where T : notnull
    {
        ReadOnlySpan<float> paramSpan = GenericArrayRoutines.ConvertToReadOnlySpan(param);
        Span<float> paramGradientSpan = GenericArrayRoutines.ConvertToSpan(paramGradient);

        Debug.Assert(paramSpan.Length == paramGradientSpan.Length, "Parameter and its gradient must have the same length.");

        ApplyGradientCore(paramSpan, paramGradientSpan);
    }

    /// <summary>
    /// Computes and returns the penalty for the specified parameter when penalty reporting is enabled.
    /// </summary>
    /// <typeparam name="T">The type of the parameter to evaluate. Must be a non-null type.</typeparam>
    /// <param name="param">The parameter value to evaluate for penalty calculation.</param>
    /// <returns>The computed penalty value when reporting is enabled; otherwise, <c>0f</c>.</returns>
    public float ReportPenalty<T>(T param) where T : notnull
    {
        if (reportPenaltyInLoss)
        {
            ReadOnlySpan<float> paramSpan = GenericArrayRoutines.ConvertToReadOnlySpan(param);
            return CalculatePenaltyCore(paramSpan);
        }
        return 0f;
    }

    /// <summary>
    /// Updates parameter values from the specified input by converting it to a <see cref="System.Span{System.Single}"/>
    /// and passing it to the core update routine.
    /// </summary>
    /// <typeparam name="T">The non-null input type that can be converted to a span of single-precision values.</typeparam>
    /// <param name="param">The input containing parameter values to apply.</param>
    public void UpdateParams<T>(T param) where T : notnull
    {
        Span<float> paramSpan = GenericArrayRoutines.ConvertToSpan(param);
        UpdateParamsCore(paramSpan);
    }
    
    /// <summary>
    /// Applies a gradient update to the specified parameter span using the corresponding gradient span.
    /// </summary>
    /// <param name="paramSpan">The span of parameter values to update.</param>
    /// <param name="paramGradientSpan">The span of gradient values corresponding to <paramref name="paramSpan"/>.</param>
    protected internal virtual void ApplyGradientCore(ReadOnlySpan<float> paramSpan, Span<float> paramGradientSpan)
    {
    }

    protected internal virtual float CalculatePenaltyCore(ReadOnlySpan<float> paramSpan) 
        => 0f;

    /// <summary>
    /// Updates parameter values in the specified span.
    /// </summary>
    /// <remarks>Override in a derived type to apply parameter updates in place.</remarks>
    /// <param name="paramSpan">A span containing the parameter values to update.</param>
    protected internal virtual void UpdateParamsCore(Span<float> paramSpan)
    {
    }
}
