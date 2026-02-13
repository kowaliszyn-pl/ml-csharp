// Neural Networks in C♯
// File name: Operation.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations;

public abstract class Operation
{
    bool _registered = false;

    public void SetRegistered()
    {
        if(_registered)
            throw new InvalidOperationException($"Operation '{this}' is already registered.");

        _registered = true;
    }

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
    private TOut? _output;

    protected TIn Input
    {
        get
        {
            Debug.Assert(_input != null, "Input must not be null here.");
            return _input;
        }
    }

    protected TOut Output
    {
        get
        {
            Debug.Assert(_output != null, "Output must not be null here.");
            return _output;
        }
    }

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
    private static void EnsureSameShapeForInput(TIn? input, TIn inputGradient)
    {
        EnsureSameShape(input, inputGradient);
    }

    [Conditional("DEBUG")]
    private static void EnsureSameShapeForOutput(TOut? output, TOut outputGradient)
    {
        EnsureSameShape(output, outputGradient);
    }

    /// <summary>
    /// Computes output.
    /// </summary>
    protected abstract TOut CalcOutput(bool inference);

    /// <summary>
    /// Calculates input gradient.
    /// </summary>
    /// <remarks>
    /// Based on outputGradient, calculates changes in input.
    /// </remarks>
    protected abstract TIn CalcInputGradient(TOut outputGradient);

    public override object Forward(object input, bool inference) => Forward((TIn)input, inference);

    public override object Backward(object outputGradient) => Backward((TOut)outputGradient);

    public override Type GetOutputType() => typeof(TOut);

    public override Type GetInputType() => typeof(TIn);
}
