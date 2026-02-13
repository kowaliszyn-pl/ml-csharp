// Neural Networks in C♯
// File name: ParamOperation4D.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Utils.ModelUtils;

namespace NeuralNetworks.Operations.Parameterized;

public abstract class ParamOperation4D : Operation<float[,,,], float[,,,]>, IParamOperation
{
    internal abstract int GetParamCount();
    internal abstract void UpdateParams(Layer? layer, Optimizer optimizer);
    internal abstract ParamOperationData GetData();
    internal abstract void ApplyData(ParamOperationData data, int layerIndex, int operationIndex);

    void IParamOperation.ApplyData(ParamOperationData data, int layerIndex, int operationIndex)
        => ApplyData(data, layerIndex, operationIndex);

    ParamOperationData IParamOperation.GetData() => GetData();
    int IParamOperation.GetParamCount() => GetParamCount();
    void IParamOperation.UpdateParams(Layer? layer, Optimizer optimizer) => UpdateParams(layer, optimizer);
}

/// <summary>
/// An Operation with parameters of type TParam.
/// </summary>
/// <param name="param">Parameter matrix.</param>
public abstract class ParamOperation4D<TParam>(TParam param) : ParamOperation4D
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
}
