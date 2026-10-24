// Neural Networks in C♯
// File name: L1Regularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

/// <summary>
/// Applies L1 regularization by adding a penalty proportional to the sum of absolute parameter values.
/// </summary>
/// <remarks>Uses a positive regularization coefficient to encourage sparse model parameters. During optimization,
/// it adds l1 * sign(parameter) to each parameter gradient and computes the regularization penalty as l1 multiplied by
/// the L1 norm of the parameter vector.</remarks>
public class L1Regularizer : BaseRegularizer
{

    private readonly float _l1;

    /// <summary>
    /// Initializes a new instance of the <see cref="L1Regularizer"/> class with the specified L1 regularization factor.
    /// </summary>
    /// <param name="l1">Positive L1 regularization factor.</param>
    /// <param name="reportPenaltyInLoss">Indicates whether to include the regularization penalty in the reported loss value.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="l1"/> is less than or equal to 0.</exception>
    public L1Regularizer(float l1, bool reportPenaltyInLoss = false) : base(reportPenaltyInLoss)
    {
        if (l1 <= 0f)
            throw new ArgumentOutOfRangeException(nameof(l1), "L1 regularization parameter must be positive.");

        _l1 = l1;
    }

    /// <summary>
    /// Adds the L1 regularization gradient contribution for each parameter to the gradient buffer.
    /// </summary>
    /// <param name="paramSpan">A read-only span of parameter values used to compute the sign-based L1 contribution.</param>
    /// <param name="paramGradientSpan">A span of gradient values to update in place with the L1 contribution.</param>
    protected internal override void ApplyGradientCore(ReadOnlySpan<float> paramSpan, Span<float> paramGradientSpan)
    {
        for (int i = 0; i < paramSpan.Length; i++)
        {
            paramGradientSpan[i] += _l1 * MathF.Sign(paramSpan[i]);
        }
    }

    /// <summary>
    /// Calculates the L1 penalty for the specified parameter values.
    /// </summary>
    /// <param name="paramSpan">A read-only span containing the parameter values to evaluate.</param>
    /// <returns>The L1 coefficient multiplied by the sum of the absolute values in <paramref name="paramSpan"/>.</returns>
    protected internal override float CalculatePenaltyCore(ReadOnlySpan<float> paramSpan)
    {
        float sum = 0f;
        foreach (float paramValue in paramSpan)
        {
            sum += MathF.Abs(paramValue);
        }
        return _l1 * sum;
    }

    public override string ToString()
        => $"L1Regularizer(l1={_l1}, reportPenaltyInLoss={ReportPenaltyInLoss})";
}
