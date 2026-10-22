// Neural Networks in C♯
// File name: Regularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Utils.GenericArrayUtils;

namespace NeuralNetworks.Regularization;

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
public sealed class Regularizer(float l1 = 0f, float l2 = 0f, bool includeInLossValue = false)
{
    public float L1 => l1;

    public float L2 => l2;

    public bool IncludeInLossValue => includeInLossValue;

    /// <summary>
    /// Adds the L1/L2 regularization gradient contribution to <paramref name="paramGradient"/> in place.
    /// </summary>
    public void ApplyGradient<T>(T param, T paramGradient)
    {
        Span<float> paramSpan = ConvertToSpan(param);
        Span<float> paramGradientSpan = ConvertToSpan(paramGradient);

        for (int i = 0; i < paramSpan.Length; i++)
        {
            float value = paramSpan[i];
            float gradientContribution = 0f;

            if (l1 != 0f)
                gradientContribution += l1 * MathF.Sign(value);

            if (l2 != 0f)
                gradientContribution += 2f * l2 * value;

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
            if (l1 != 0f)
                l1Sum += MathF.Abs(value);

            if (l2 != 0f)
                l2Sum += value * value;
        }

        penalty += l1 * l1Sum;
        penalty += l2 * l2Sum;

        return penalty;
    }

    public override string ToString()
        => $"Regularizer (l1={l1}, l2={l2}, includeInLossValue={includeInLossValue})";
}
