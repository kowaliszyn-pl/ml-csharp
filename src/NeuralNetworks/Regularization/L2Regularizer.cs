// Neural Networks in C♯
// File name: L2Regularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

/// <summary>
/// Applies L2 regularization by adding a penalty proportional to the squared magnitude of model parameters.
/// </summary>
/// <remarks>Uses the L2 penalty form l2 * Σ(w^2), contributes 2 * l2 * w to each parameter gradient,
/// and can optionally include the penalty term in the reported loss via the base regularizer configuration.</remarks>
public class L2Regularizer : BaseRegularizer
{
    private readonly float _l2;

    /// <summary>
    /// Initializes a new instance of the <c>L2Regularizer</c> class with the specified L2 regularization factor.
    /// </summary>
    /// <param name="l2">Positive L2 regularization factor.</param>
    /// <param name="reportPenaltyInLoss">Indicates whether to include the regularization penalty in the reported loss.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="l2"/> is less than or equal to 0.</exception>
    public L2Regularizer(float l2, bool reportPenaltyInLoss = false) : base(reportPenaltyInLoss)
    {
        if (l2 <= 0f)
            throw new ArgumentOutOfRangeException(nameof(l2), "L2 regularization parameter must be positive.");

        _l2 = l2;
    }

    /// <summary>
    /// Adds the L2 regularization gradient contribution for each parameter element to the existing gradient values.
    /// </summary>
    /// <remarks>For each index, adds <c>2 * _l2 * paramSpan[i]</c> to <c>paramGradientSpan[i]</c>.</remarks>
    /// <param name="paramSpan">A read-only span containing the parameter values used to compute the L2 gradient term.</param>
    /// <param name="paramGradientSpan">A span containing gradient values to update in place with the L2 contribution.</param>
    protected internal override void ApplyGradientCore(ReadOnlySpan<float> paramSpan, Span<float> paramGradientSpan)
    {
        for (int i = 0; i < paramSpan.Length; i++)
        {
            paramGradientSpan[i] += 2f * _l2 * paramSpan[i];
        }
    }

    /// <summary>
    /// Computes the L2 penalty as the scaled sum of squared parameter values.
    /// </summary>
    /// <param name="paramSpan">A read-only span of parameter values to penalize.</param>
    /// <returns>The L2 coefficient multiplied by the sum of squares of all values in <paramref name="paramSpan"/>.</returns>
    protected internal override float CalculatePenaltyCore(ReadOnlySpan<float> paramSpan)
    {
        float sum = 0f;
        foreach (float paramValue in paramSpan)
        {
            sum += paramValue * paramValue;
        }
        return _l2 * sum;
    }

    public override string ToString()
        => $"L2Regularizer(l2={_l2}, reportPenaltyInLoss={ReportPenaltyInLoss})";
}
