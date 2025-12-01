// Machine Learning Utils
// File name: Operation.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using System.Diagnostics;

using MachineLearning.NeuralNetwork.Exceptions;

namespace NeuralNetworks.Operations;

public abstract class Operation
{
    public abstract Type GetOutputType();
    public abstract Type GetInputType();
    public abstract object Forward(object input, bool inference);
    public abstract object Backward(object outputGradient);
}

/// <summary>
/// Base class for an "operation" in a neural network.
/// </summary>
public abstract class Operation<TIn, TOut> : Operation
    where TIn : notnull
    where TOut : notnull
{
    private TIn? _input;
    // private Matrix? _inputGradient; // not used - to remove
    private TOut? _output;

    protected TIn Input => _input ?? throw new NotYetCalculatedException();

    protected TOut Output => _output ?? throw new NotYetCalculatedException();

    /// <summary>
    /// Converts input to output.
    /// </summary>
    public virtual TOut Forward(TIn input, bool inference)
    {
        _input = input;
        _output = CalcOutput(inference);
        return _output;
    }

    /// <summary>
    /// Converts output gradient to input gradient.
    /// </summary>
    public virtual TIn Backward(TOut outputGradient)
    {
        EnsureSameShapeForOutput(_output, outputGradient);
        TIn inputGradient = CalcInputGradient(outputGradient);

        EnsureSameShapeForInput(_input, inputGradient);
        return inputGradient;
    }

    [Conditional("DEBUG")]
    protected abstract void EnsureSameShapeForInput(TIn? input, TIn inputGradient);

    [Conditional("DEBUG")]
    protected abstract void EnsureSameShapeForOutput(TOut? output, TOut outputGradient);

    /// <summary>
    /// Computes output.
    /// </summary>
    protected abstract TOut CalcOutput(bool inference);

    /// <summary>
    /// Calculates input gradient.
    /// </summary>
    /// <remarks>
    /// Na podstawie outputGradient oblicza zmiany w input.
    /// </remarks>
    protected abstract TIn CalcInputGradient(TOut outputGradient);

    public override object Forward(object input, bool inference) => Forward((TIn)input, inference);

    public override object Backward(object outputGradient) => Backward((TOut)outputGradient);

    public override Type GetOutputType() => typeof(TOut);

    public override Type GetInputType() => typeof(TIn);

    #region Clone

    protected virtual Operation<TIn, TOut> CloneBase()
    {
        Operation<TIn, TOut> clone = (Operation<TIn, TOut>)MemberwiseClone();
        // TODO:
        //clone._input = _input?.Clone();
        // TODO:
        //clone._output = _output?.Clone();
        return clone;
    }

    public Operation<TIn, TOut> Clone() => CloneBase();

    #endregion
}
