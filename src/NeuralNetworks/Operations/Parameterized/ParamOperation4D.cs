// Machine Learning Utils
// File name: ParamOperation4D.cs
// Code It Yourself with .NET, 2024

using System;
using System.Diagnostics;

using NeuralNetworks.Layers;
using NeuralNetworks.Operations;
using NeuralNetworks.Operations.Interfaces;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Operations.Parameterized;

public abstract class ParamOperation4D : Operation4D, IParameterCountProvider, IParameterUpdater
{
    public abstract int GetParamCount();
    public abstract void UpdateParams(Layer? layer, Optimizer optimizer);
}

/// <summary>
/// An Operation with parameters of type TParam.
/// </summary>
/// <param name="param">Parameter matrix.</param>
public abstract class ParamOperation4D<TParam>(TParam param) : ParamOperation4D, IParameterStateProvider
    where TParam : class
{
    private TParam? _paramGradient;

    protected TParam Param => param;

    internal TParam ParamGradient
    {
        get
        {
            Debug.Assert(_paramGradient != null, "ParamGradient must not be null here.");
            return _paramGradient;
        }
    }

    public override float[,,,] Backward(float[,,,] outputGradient)
    {
        float[,,,] inputGrad = base.Backward(outputGradient);

        _paramGradient = CalcParamGradient(outputGradient);
        EnsureSameShapeForParam(param, _paramGradient);
        return inputGrad;
    }

    [Conditional("DEBUG")]
    protected abstract void EnsureSameShapeForParam(TParam? param, TParam paramGradient);

    protected abstract TParam CalcParamGradient(float[,,,] outputGradient);

    protected override Operation<float[,,,], float[,,,]> CloneBase()
    {
        ParamOperation4D<TParam> clone = (ParamOperation4D<TParam>)base.CloneBase();
        //clone._paramGradient = _paramGradient?.Clone();
        return clone;
    }

    ParameterSnapshot IParameterStateProvider.Capture()
    {
        if (param is not Array array)
            throw new NotSupportedException($"Operation '{GetType().Name}' stores unsupported parameter type '{typeof(TParam)}'.");

        return ParameterSnapshot.FromArray(array);
    }

    void IParameterStateProvider.Restore(ParameterSnapshot snapshot)
    {
        if (param is not Array array)
            throw new NotSupportedException($"Operation '{GetType().Name}' stores unsupported parameter type '{typeof(TParam)}'.");

        snapshot.CopyTo(array);
    }
}
