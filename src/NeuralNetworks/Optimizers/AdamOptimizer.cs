// Neural Networks in C♯
// File name: AdamOptimizer.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;

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
    private readonly float _epsilon;

    private readonly Dictionary<float[], State1D> _states1D = [];
    private readonly Dictionary<float[,], State2D> _states2D = [];
    private readonly Dictionary<float[,,,], State4D> _states4D = [];

    public AdamOptimizer(LearningRate learningRate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : base(learningRate)
    {
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
    }

    public override void Update(Layer? layer, float[] param, float[] paramGradient)
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

        Debug.Assert(param.HasSameShape(paramGradient));

        (int t, float[] m, float[] v) = GetOrCreateState(param);

        float beta1t = MathF.Pow(_beta1, t);
        float beta2t = MathF.Pow(_beta2, t);

        float lr = LearningRate.GetLearningRate();
        WarnIfLearningRateIsSuspicious(lr);

        int length = param.Length;

        for (int i = 0; i < length; i++)
        {
            m[i] = _beta1 * m[i] + (1 - _beta1) * paramGradient[i];
            v[i] = _beta2 * v[i] + (1 - _beta2) * paramGradient[i] * paramGradient[i];

            float mHat = m[i] / (1 - beta1t);
            float vHat = v[i] / (1 - beta2t);

            param[i] -= lr * mHat / (MathF.Sqrt(vHat) + _epsilon);
        }
    }

    public override void Update(Layer? layer, float[,] param, float[,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        (int t, float[,] m, float[,] v) = GetOrCreateState(param);

        float beta1t = MathF.Pow(_beta1, t);
        float beta2t = MathF.Pow(_beta2, t);

        float lr = LearningRate.GetLearningRate();
        WarnIfLearningRateIsSuspicious(lr);

        int dim1 = param.GetLength(0);
        int dim2 = param.GetLength(1);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                m[i, j] = _beta1 * m[i, j] + (1 - _beta1) * paramGradient[i, j];
                v[i, j] = _beta2 * v[i, j] + (1 - _beta2) * paramGradient[i, j] * paramGradient[i, j];

                float mHat = m[i, j] / (1 - beta1t);
                float vHat = v[i, j] / (1 - beta2t);

                param[i, j] -= lr * mHat / (MathF.Sqrt(vHat) + _epsilon);
            }
        }
    }

    public override void Update(Layer? layer, float[,,,] param, float[,,,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        (int t, float[,,,] m, float[,,,] v) = GetOrCreateState(param);
        float beta1t = MathF.Pow(_beta1, t);
        float beta2t = MathF.Pow(_beta2, t);

        float lr = LearningRate.GetLearningRate();
        WarnIfLearningRateIsSuspicious(lr);

        int dim1 = param.GetLength(0);
        int dim2 = param.GetLength(1);
        int dim3 = param.GetLength(2);
        int dim4 = param.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        m[i, j, k, l] = _beta1 * m[i, j, k, l] + (1 - _beta1) * paramGradient[i, j, k, l];
                        v[i, j, k, l] = _beta2 * v[i, j, k, l] + (1 - _beta2) * paramGradient[i, j, k, l] * paramGradient[i, j, k, l];

                        float mHat = m[i, j, k, l] / (1 - beta1t);
                        float vHat = v[i, j, k, l] / (1 - beta2t);

                        param[i, j, k, l] -= lr * mHat / (MathF.Sqrt(vHat) + _epsilon);
                    }
                }
            }
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
        var newState = new State1D(param);
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
        var newState = new State2D(param);
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
        var newState = new State4D(param);
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
        => $"Adam (learningRate={LearningRate}, beta1={_beta1}, beta2={_beta2}, epsilon={_epsilon})";

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

    private sealed class State2D
    {
        public int T { get; set; } = 1;
        public float[,] M { get; }
        public float[,] V { get; }

        public State2D(float[,] param)
        {
            int rows = param.GetLength(0);
            int cols = param.GetLength(1);
            M = new float[rows, cols];
            V = new float[rows, cols];
        }

        public void Deconstruct(out int t, out float[,] m, out float[,] v)
        {
            t = T;
            m = M;
            v = V;
        }
    }

    private sealed class State4D
    {
        public int T { get; set; } = 1;
        public float[,,,] M { get; }
        public float[,,,] V { get; }
        public State4D(float[,,,] param)
        {
            int dim1 = param.GetLength(0);
            int dim2 = param.GetLength(1);
            int dim3 = param.GetLength(2);
            int dim4 = param.GetLength(3);
            M = new float[dim1, dim2, dim3, dim4];
            V = new float[dim1, dim2, dim3, dim4];
        }
        public void Deconstruct(out int t, out float[,,,] m, out float[,,,] v)
        {
            t = T;
            m = M;
            v = V;
        }
    }
}