// Neural Networks in C♯
// File name: Regularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using static NeuralNetworks.Core.GenericArrayRoutines;

namespace NeuralNetworks.Regularization;

/// <summary>
/// Represents L1 and L2 regularization settings (elastic net) for neural network parameters and provides methods to
/// apply regularization to gradients and compute the corresponding penalty.
/// </summary>
/// <remarks>Use <c>l1</c> and <c>l2</c> coefficients to enable L1, L2, or combined regularization. When
/// <c>IncludeInLossValue</c> is <see langword="true"/>, the regularization penalty is intended to be included in the
/// reported training loss; otherwise, regularization affects gradients without changing the reported loss
/// value.</remarks>
public sealed class Regularizer
{
    private readonly float _l1, _l2;
    private readonly bool _includeInLossValue;

    /// <summary>
    /// Configures L1 and/or L2 regularization (elastic net) applied to a parameter (weights or biases) of a
    /// neural network operation.
    /// </summary>
    /// <param name="l1">L1 regularization coefficient. Use 0 to disable L1 regularization.</param>
    /// <param name="l2">L2 regularization coefficient. Use 0 to disable L2 regularization.</param>
    /// <param name="includeInLossValue">
    /// When <see langword="true"/>, the regularization penalty for this parameter is added to the reported training loss.
    /// When <see langword="false"/>, only the gradient (weight decay) is affected and the reported loss is unchanged.
    /// </param>
    public Regularizer(float l1 = 0f, float l2 = 0f, bool includeInLossValue = false)
    {
        Debug.Assert(l1 >= 0f, "L1 regularization coefficient must be non-negative.");
        Debug.Assert(l2 >= 0f, "L2 regularization coefficient must be non-negative.");

        _l1 = l1;
        _l2 = l2;
        _includeInLossValue = includeInLossValue;
    }

    public float L1 => _l1;

    public float L2 => _l2;

    public bool IncludeInLossValue => _includeInLossValue;

    /// <summary>
    /// Adds the L1/L2 regularization gradient contribution to <paramref name="paramGradient"/> in place.
    /// </summary>
    public void ApplyGradient<T>(T param, T paramGradient)
    {
        Span<float> paramSpan = ConvertToSpan(param);
        Span<float> paramGradientSpan = ConvertToSpan(paramGradient);

        Debug.Assert(paramSpan.Length == paramGradientSpan.Length, "Parameter and its gradient must have the same length.");

        for (int i = 0; i < paramSpan.Length; i++)
        {
            float value = paramSpan[i];
            float gradientContribution = 0f;

            if (_l1 != 0f)
                gradientContribution += _l1 * MathF.Sign(value);

            if (_l2 != 0f)
                gradientContribution += 2f * _l2 * value;

            paramGradientSpan[i] += gradientContribution;
        }
    }

    /// <summary>
    /// Calculates the L1/L2 regularization penalty for <paramref name="param"/>.
    /// </summary>
    public float CalculatePenalty<T>(T param)
    {
        Span<float> paramSpan = ConvertToSpan(param);

        float penalty = 0f;
        float l1Sum = 0f;
        float l2Sum = 0f;

        foreach (float value in paramSpan)
        {
            if (_l1 != 0f)
                l1Sum += MathF.Abs(value);

            if (_l2 != 0f)
                l2Sum += value * value;
        }

        penalty += _l1 * l1Sum;
        penalty += _l2 * l2Sum;

        return penalty;
    }

    public override string ToString()
        => $"Regularizer (l1={_l1}, l2={_l2}, includeInLossValue={_includeInLossValue})";
}
