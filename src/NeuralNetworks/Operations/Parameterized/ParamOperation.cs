// Neural Networks in C♯
// File name: ParamOperation2D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Utils.ModelUtils;
using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations.Parameterized;

public abstract class ParamOperation<TIn, TOut> : Operation<TIn, TOut>, IParamOperation
    where TIn : notnull
    where TOut : notnull
{
    internal abstract int GetParamCount();
    internal abstract void UpdateParams(Optimizer optimizer);
    internal abstract ParamOperationData GetData();
    internal abstract void ApplyData(ParamOperationData data, int layerIndex, int operationIndex);

    void IParamOperation.ApplyData(ParamOperationData data, int layerIndex, int operationIndex)
        => ApplyData(data, layerIndex, operationIndex);

    ParamOperationData IParamOperation.GetData() => GetData();
    int IParamOperation.GetParamCount() => GetParamCount();
    void IParamOperation.UpdateParams(Optimizer optimizer) => UpdateParams(optimizer);
}

/// <summary>
/// An Operation with parameters of type TParam.
/// </summary>
/// <param name="param">Parameter matrix.</param>
public abstract class ParamOperation<TIn, TOut, TParam>(TParam param) : ParamOperation<TIn, TOut>
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
        return inputGrad;
    }

    [Conditional("DEBUG")]
    private static void EnsureSameShapeForParam(TParam? param, TParam paramGradient)
    {
        EnsureSameShape(param, paramGradient);
    }

    protected abstract TParam CalcParamGradient(TOut outputGradient);

    internal override ParamOperationData GetData()
    {
        if (param is not Array array)
            throw new NotSupportedException($"Operation '{GetType().Name}' stores unsupported parameter type '{typeof(TParam)}'.");

        string operationType = GetTypeIdentifier(GetType());

        return new ParamOperationData(operationType, ParamOperationParams.FromArray(array));
    }

    internal override void ApplyData(ParamOperationData data, int layerIndex, int operationIndex)
    {
        if (param is not Array array)
            throw new NotSupportedException($"Operation '{GetType().Name}' stores unsupported parameter type '{typeof(TParam)}'.");

        EnsureTypeMatch(data.OperationType, GetType(), layerIndex, operationIndex);

        data.Parameters.CopyTo(array);
    }

    internal override int GetParamCount()
    {
        if (param is Array array)
            return array.Length;
        else
            throw new NotSupportedException();
    }

    internal override void UpdateParams(Optimizer optimizer)
    {
        optimizer.Update(param, ParamGradient);
    }
}