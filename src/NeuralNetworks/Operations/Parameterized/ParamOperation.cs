// Neural Networks in C♯
// File name: ParamOperation.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Optimizers;
using NeuralNetworks.Regularization;

using static NeuralNetworks.Core.GenericArrayRoutines;
using static NeuralNetworks.Utils.ModelUtils;

namespace NeuralNetworks.Operations.Parameterized;

public abstract class ParamOperation<TIn, TOut> : Operation<TIn, TOut>, IParamOperation
    where TIn : notnull
    where TOut : notnull
{
    public abstract int GetParamCount();
    public abstract void UpdateParams(Optimizer optimizer);
    public abstract ParamOperationData GetData();
    public abstract void ApplyData(ParamOperationData data, int layerIndex, int operationIndex);
    public abstract float GetRegularizationLoss();
}

/// <summary>
/// An Operation with parameters of type TParam.
/// </summary>
/// <param name="param">Parameter matrix.</param>
/// <param name="regularizer">Optional L1/L2 regularizer applied to this parameter's gradient (and optionally its loss).</param>
public abstract class ParamOperation<TIn, TOut, TParam>(TParam param, Regularizer? regularizer = null) : ParamOperation<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
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

    public override TIn Backward(TOut outputGradient)
    {
        TIn inputGrad = base.Backward(outputGradient);

        _paramGradient = CalcParamGradient(outputGradient);
        EnsureSameShapeForParam(param, _paramGradient);

        regularizer?.ApplyGradient(param, _paramGradient);

        return inputGrad;
    }

    public override float GetRegularizationLoss()
        => regularizer is { IncludeInLossValue: true }
            ? regularizer.CalculatePenalty(param)
            : 0f;

    [Conditional("DEBUG")]
    private static void EnsureSameShapeForParam(TParam? param, TParam paramGradient) 
        => EnsureSameShape(param, paramGradient);

    protected abstract TParam CalcParamGradient(TOut outputGradient);

    public override ParamOperationData GetData()
    {
        if (param is not Array array)
            throw new NotSupportedException($"Operation '{GetType().Name}' stores unsupported parameter type '{typeof(TParam)}'.");

        string operationType = GetTypeIdentifier(GetType());

        return new ParamOperationData(operationType, ParamOperationParams.FromArray(array));
    }

    public override void ApplyData(ParamOperationData data, int layerIndex, int operationIndex)
    {
        if (param is not Array array)
            throw new NotSupportedException($"Operation '{GetType().Name}' stores unsupported parameter type '{typeof(TParam)}'.");

        EnsureTypeMatch(data.OperationType, GetType(), layerIndex, operationIndex);

        data.Parameters.CopyTo(array);
    }

    public override int GetParamCount()
    {
        if (param is Array array)
            return array.Length;
        else
            throw new NotSupportedException();
    }

    public override void UpdateParams(Optimizer optimizer) 
        => optimizer.Update(param, ParamGradient);
}