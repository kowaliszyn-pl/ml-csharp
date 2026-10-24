// Neural Networks in C♯
// File name: MaxNormRegularizer.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Regularization;

/// <summary>
/// Applies max-norm regularization by scaling a parameter vector when its L2 norm exceeds a specified limit.
/// </summary>
/// <remarks>If the vector norm is greater than the configured limit, all elements are scaled by the same factor
/// so the resulting norm equals the limit.</remarks>
public class MaxNormRegularizer : BaseRegularizer
{
    private readonly float _maxNorm;

    /// <param name="maxNorm">The maximum allowed L2 norm for the parameter vector.</param>
    public MaxNormRegularizer(float maxNorm) : base(false)
    {
        if(maxNorm <= 0f)
            throw new ArgumentOutOfRangeException(nameof(maxNorm), "Max norm must be positive.");

        _maxNorm = maxNorm;
    }

    /// <summary>
    /// Scales the parameter vector in place when its Euclidean norm exceeds the configured maximum norm.
    /// </summary>
    /// <remarks>Computes the L2 norm of the span and applies a uniform scaling factor so the resulting norm
    /// equals the maximum norm while preserving the vector direction.</remarks>
    /// <param name="paramSpan">A span containing the parameter values to evaluate and, if needed, scale.</param>
    protected internal override void UpdateParamsCore(Span<float> paramSpan)
    {
        float normSq = 0f;
        foreach (float v in paramSpan)
        {
            normSq += v * v;
        }
        float norm = MathF.Sqrt(normSq);

        if (norm > _maxNorm)
        {
            float scale = _maxNorm / norm;
            for (int i = 0; i < paramSpan.Length; i++)
            {
                paramSpan[i] *= scale;
            }
        }
    }

    public override string ToString() 
        => $"MaxNormRegularizer(MaxNorm={_maxNorm})";
}
