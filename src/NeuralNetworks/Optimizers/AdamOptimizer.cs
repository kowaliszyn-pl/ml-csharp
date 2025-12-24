// Neural Networks in C♯
// File name: AdamOptimizer.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
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

    private readonly Dictionary<float[], State1D> _states1D = [];
    private readonly Dictionary<float[,], State2D> _states2D = [];
    private readonly Dictionary<float[,,,], State4D> _states4D = [];

    public AdamOptimizer(LearningRate learningRate, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : base(learningRate)
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
    }

    public override void Update(Layer? layer, float[] param, float[] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        (int t, float[] m, float[] v) = GetOrCreateState(param);

        ApplyUpdate(
            param.AsSpan(),
            paramGradient.AsSpan(),
            v.AsSpan(),
            m.AsSpan(),
            t
        );
    }

    public override void Update(Layer? layer, float[,] param, float[,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        (int t, float[,] m, float[,] v) = GetOrCreateState(param);

        ApplyUpdate(
            MemoryMarshal.CreateSpan(ref param[0, 0], param.Length),
            MemoryMarshal.CreateReadOnlySpan(ref paramGradient[0, 0], param.Length),
            MemoryMarshal.CreateSpan(ref v[0, 0], param.Length),
            MemoryMarshal.CreateSpan(ref m[0, 0], param.Length),
            t
        );
    }

    public override void Update(Layer? layer, float[,,,] param, float[,,,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        (int t, float[,,,] m, float[,,,] v) = GetOrCreateState(param);

        ApplyUpdate(
            MemoryMarshal.CreateSpan(ref param[0, 0, 0, 0], param.Length),
            MemoryMarshal.CreateReadOnlySpan(ref paramGradient[0, 0, 0, 0], param.Length),
            MemoryMarshal.CreateSpan(ref v[0, 0, 0, 0], param.Length),
            MemoryMarshal.CreateSpan(ref m[0, 0, 0, 0], param.Length),
            t
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyUpdate(Span<float> param, ReadOnlySpan<float> paramGradient, Span<float> v, Span<float> m, int t)
    {
        /*
         * The process of updating parameters using Adam optimizer involves the following steps:
         * 1. Compute the first moment (m) and second moment (v) estimates.
         * 2. Update biased first moment estimate.
         * 3. Update biased second moment estimate.
         * 4. Compute bias-corrected first moment estimate.
         * 5. Compute bias-corrected second moment estimate.
         * 6. Update parameters using the Adam update rule: param -= learningRate * mHat / (sqrt(vHat) + epsilon), 
         *    where epsilon is a small constant to prevent division by zero, mHat is the bias-corrected first moment estimate,
         *    and vHat is the bias-corrected second moment estimate.
         */

        float lr = LearningRate.GetLearningRate();
        WarnIfLearningRateIsSuspicious(lr);

        float beta1t = MathF.Pow(_beta1, t);
        float beta2t = MathF.Pow(_beta2, t);

        for (int i = 0; i < param.Length; i++)
        {
            m[i] = _beta1 * m[i] + (1 - _beta1) * paramGradient[i];
            v[i] = _beta2 * v[i] + (1 - _beta2) * paramGradient[i] * paramGradient[i];
            float mHat = m[i] / (1 - beta1t);
            float vHat = v[i] / (1 - beta2t);
            param[i] -= lr * mHat / (MathF.Sqrt(vHat) + _eps);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private State1D GetOrCreateState(float[] param)
    {
        if (_states1D.TryGetValue(param, out State1D? state))
        {
            state.T++;
            return state;
        }
        State1D newState = new(param);
        _states1D[param] = newState;
        return newState;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private State2D GetOrCreateState(float[,] param)
    {
        if (_states2D.TryGetValue(param, out State2D? state))
        {
            state.T++;
            return state;
        }
        State2D newState = new(param);
        _states2D[param] = newState;
        return newState;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private State4D GetOrCreateState(float[,,,] param)
    {
        if (_states4D.TryGetValue(param, out State4D? state))
        {
            state.T++;
            return state;
        }
        State4D newState = new(param);
        _states4D[param] = newState;
        return newState;
    }

    static bool s_warningEmitted = false;

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

    public override string ToString()
        => $"Adam (learningRate={LearningRate}, beta1={_beta1}, beta2={_beta2}, eps={_eps})";

    private sealed class State1D(float[] param)
    {
        public int T { get; set; } = 1;
        public float[] M { get; } = new float[param.Length];
        public float[] V { get; } = new float[param.Length];

        public void Deconstruct(out int t, out float[] m, out float[] v)
        {
            t = T;
            m = M;
            v = V;
        }
    }

    private sealed class State2D(float[,] param)
    {
        public int T { get; set; } = 1;
        public float[,] M { get; } = param.AsZeros();
        public float[,] V { get; } = param.AsZeros();

        public void Deconstruct(out int t, out float[,] m, out float[,] v)
        {
            t = T;
            m = M;
            v = V;
        }
    }

    private sealed class State4D(float[,,,] param)
    {
        public int T { get; set; } = 1;
        public float[,,,] M { get; } = param.AsZeros();
        public float[,,,] V { get; } = param.AsZeros();

        public void Deconstruct(out int t, out float[,,,] m, out float[,,,] v)
        {
            t = T;
            m = M;
            v = V;
        }
    }
}