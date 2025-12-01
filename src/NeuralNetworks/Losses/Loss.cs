// Machine Learning Utils
// File name: Loss.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using System.Diagnostics;

using MachineLearning.NeuralNetwork.Exceptions;

namespace NeuralNetworks.Losses;

/// <summary>
/// The "loss" of a neural network.
/// </summary>
public abstract class Loss<TPrediction>
{
    private TPrediction? _prediction;
    private TPrediction? _target;

    public TPrediction Prediction => _prediction ?? throw new NotYetCalculatedException();

    protected internal TPrediction Target => _target ?? throw new NotYetCalculatedException();

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
    protected abstract void EnsureSameShape(TPrediction? prediction, TPrediction target);

    #endregion EnsureSameShape
}
