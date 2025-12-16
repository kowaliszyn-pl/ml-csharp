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
    private int _t = 0;

    private readonly Dictionary<float[], (float[] m, float[] v)> _moments1D = [];
    private readonly Dictionary<float[,], (float[,] m, float[,] v)> _moments2D = [];
    private readonly Dictionary<float[,,,], (float[,,,] m, float[,,,] v)> _moments4D = [];

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
        _t++;

        (float[]? m, float[]? v) = GetOrCreateMoments(param);

        float lr = LearningRate.GetLearningRate();
        int length = param.Length;
        float beta1t = MathF.Pow(_beta1, _t);
        float beta2t = MathF.Pow(_beta2, _t);

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
        _t++;

        (float[,]? m, float[,]? v) = GetOrCreateMoments(param);

        float lr = LearningRate.GetLearningRate();
        int dim1 = param.GetLength(0);
        int dim2 = param.GetLength(1);
        float beta1t = MathF.Pow(_beta1, _t);
        float beta2t = MathF.Pow(_beta2, _t);

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
        _t++;

        (float[,,,]? m, float[,,,]? v) = GetOrCreateMoments(param);

        float lr = LearningRate.GetLearningRate();
        int dim1 = param.GetLength(0);
        int dim2 = param.GetLength(1);
        int dim3 = param.GetLength(2);
        int dim4 = param.GetLength(3);
        float beta1t = MathF.Pow(_beta1, _t);
        float beta2t = MathF.Pow(_beta2, _t);

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
    private (float[] m, float[] v) GetOrCreateMoments(float[] param)
    {
        if (_moments1D.TryGetValue(param, out (float[] m, float[] v) moments))
        {
            return moments;
        }
        var m = new float[param.Length];
        var v = new float[param.Length];
        _moments1D.Add(param, (m, v));
        return (m, v);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private (float[,] m, float[,] v) GetOrCreateMoments(float[,] param)
    {
        if (_moments2D.TryGetValue(param, out (float[,] m, float[,] v) moments))
        {
            return moments;
        }
        var m = new float[param.GetLength(0), param.GetLength(1)];
        var v = new float[param.GetLength(0), param.GetLength(1)];
        _moments2D.Add(param, (m, v));
        return (m, v);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private (float[,,,] m, float[,,,] v) GetOrCreateMoments(float[,,,] param)
    {
        if (_moments4D.TryGetValue(param, out (float[,,,] m, float[,,,] v) moments))
        {
            return moments;
        }
        var m = new float[param.GetLength(0), param.GetLength(1), param.GetLength(2), param.GetLength(3)];
        var v = new float[param.GetLength(0), param.GetLength(1), param.GetLength(2), param.GetLength(3)];
        _moments4D.Add(param, (m, v));
        return (m, v);
    }

    public override string ToString()
        => $"Adam (learningRate={LearningRate}, beta1={_beta1}, beta2={_beta2}, epsilon={_epsilon})";
}