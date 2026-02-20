// Neural Networks in C♯
// File name: Loss.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using NeuralNetworks.Core;

namespace NeuralNetworks.Losses;

/// <summary>
/// The "loss" of a neural network.
/// </summary>
public abstract class Loss<TPrediction>
{
    private TPrediction? _prediction;
    private TPrediction? _target;

    public TPrediction Prediction
    {
        get
        {
            Debug.Assert(_prediction != null, "Prediction must not be null here.");
            return _prediction;
        }
    }

    protected internal TPrediction Target
    {
        get
        {
            Debug.Assert(_target != null, "Target must not be null here.");
            return _target;
        }
    }

    /// <summary>
    /// Computes the actual loss value
    /// </summary>
    public float Forward(TPrediction prediction, TPrediction target)
    {
        EnsureSameShape(prediction, target);
        _prediction = prediction;
        _target = target;

        return CalculateLoss();
    }

    /// <summary>
    /// Computes gradient of the loss value with respect to the input to the loss function.
    /// </summary>
    public TPrediction Backward()
    {
        TPrediction lossGradient = CalculateLossGradient();
        EnsureSameShape(_prediction, lossGradient);
        return lossGradient;
    }

    protected abstract float CalculateLoss();

    protected abstract TPrediction CalculateLossGradient();

    #region Clone

    protected virtual Loss<TPrediction> CloneBase()
    {
        Loss<TPrediction> clone = (Loss<TPrediction>)MemberwiseClone();
        // TODO: clone
        // clone._prediction = _prediction?.Clone();
        // clone._target = _target?.Clone();
        return clone;
    }

    public Loss<TPrediction> Clone() => CloneBase();

    #endregion Clone

    #region EnsureSameShape

    [Conditional("DEBUG")]
    private void EnsureSameShape(TPrediction? prediction, TPrediction target)
        => GenericUtils.EnsureSameShape(prediction, target);

    #endregion EnsureSameShape
}
