// Neural Networks in C♯
// File name: AdamOptimizer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.LearningRates;

namespace NeuralNetworks.Optimizers;

/// <summary>
/// Implements the Adam optimizer for neural network training.
/// Adam combines momentum and adaptive learning rates for each parameter.
/// </summary>
public class AdamOptimizer : Optimizer
{
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;

    private readonly Dictionary<object, State> _states = [];

    public AdamOptimizer(LearningRate learningRate, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : base(learningRate)
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
    }

    private static bool s_warningEmitted = false;

    [Conditional("DEBUG")]
    private static void WarnIfLearningRateIsSuspicious(float lr)
    {
        if (!s_warningEmitted)
        {
            // Adam typically works best with lr in [0.0001, 0.01]
            if (lr > 0.05f)
            {
                Debug.WriteLine($"[AdamOptimizer WARNING] Learning rate {lr} is unusually high for Adam. Typical values are 0.0001–0.01.");
                s_warningEmitted = true;
            }
            else if (lr < 1e-5f)
            {
                Debug.WriteLine($"[AdamOptimizer WARNING] Learning rate {lr} is very low and may cause slow convergence.");
                s_warningEmitted = true;
            }
        }
    }

    /// <summary>
    /// Updates the specified parameters in place using the Adam optimization algorithm and the provided gradients.
    /// </summary>
    /// <remarks>This method applies the Adam optimizer update rule to the parameters, maintaining
    /// per-parameter first and second moment estimates across calls. The optimizer state is tracked per <paramref
    /// name="paramsKey"/>. The method expects that <paramref name="paramsToUpdate"/> and <paramref
    /// name="paramGradients"/> have the same length; otherwise, a debug assertion will fail. The update is performed in
    /// place, modifying the values in <paramref name="paramsToUpdate"/> directly.</remarks>
    /// <param name="paramsKey">An object that uniquely identifies the parameter set to update. Used to maintain optimizer state for each
    /// parameter group.</param>
    /// <param name="paramsToUpdate">A span containing the parameter values to be updated. The values are modified in place based on the computed
    /// Adam update.</param>
    /// <param name="paramGradients">A read-only span containing the gradients corresponding to each parameter in <paramref name="paramsToUpdate"/>.
    /// Must have the same length as <paramref name="paramsToUpdate"/>.</param>
    protected override void Update(object paramsKey, Span<float> paramsToUpdate, ReadOnlySpan<float> paramGradients)
    {
        int length = paramsToUpdate.Length;
        Debug.Assert(length == paramGradients.Length, "Parameter and gradient spans must have the same length.");

        float learningRate = LearningRate.GetLearningRate();
        WarnIfLearningRateIsSuspicious(learningRate);

        // Get or create (t, v, m) for the given parameter key. This will be used to store the first/second momentum term for each parameter.

        if (_states.TryGetValue(paramsKey, out State? currentState))
        {
            currentState.T++;
        }
        else
        {
            currentState = new(length);
            _states[paramsKey] = currentState;
        }

        Debug.Assert(currentState.M.Length == length && currentState.V.Length == length, "State vectors must match parameter length.");

        /*
       * The process of updating parameters using Adam optimizer involves the following steps:
       * 1. Compute the first moment (mean, m) and second moment (variance, v) estimates.
       * 2. Update biased first moment estimate.
       * 3. Update biased second moment estimate.
       * 4. Compute bias-corrected first moment estimate.
       * 5. Compute bias-corrected second moment estimate.
       * 6. Update parameters using the Adam update rule: param -= learningRate * mHat / (sqrt(vHat) + epsilon), 
       *    where epsilon is a small constant to prevent division by zero, mHat is the bias-corrected first moment estimate,
       *    and vHat is the bias-corrected second moment estimate.
       */

        float beta1t = MathF.Pow(_beta1, currentState.T);
        float beta2t = MathF.Pow(_beta2, currentState.T);

        for (int i = 0; i < length; i++)
        {
            float paramGrad = paramGradients[i];
            float mean = _beta1 * currentState.M[i] + (1 - _beta1) * paramGrad; // first moment
            float variance = _beta2 * currentState.V[i] + (1 - _beta2) * paramGrad * paramGrad; // second moment
            float mHat = mean / (1 - beta1t);
            float vHat = variance / (1 - beta2t);
            currentState.M[i] = mean;
            currentState.V[i] = variance;
            paramsToUpdate[i] -= learningRate * mHat / (MathF.Sqrt(vHat) + _eps);
        }
    }

    public override string ToString()
        => $"Adam (learningRate={LearningRate}, beta1={_beta1}, beta2={_beta2}, eps={_eps})";


    private sealed class State(int paramLength)
    {
        public int T { get; set; } = 1;
        public float[] M { get; } = new float[paramLength];
        public float[] V { get; } = new float[paramLength];

        public void Deconstruct(out int t, out float[] m, out float[] v)
        {
            t = T;
            m = M;
            v = V;
        }
    }
}