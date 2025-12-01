// Machine Learning Utils
// File name: ParamOperation2D.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using System.Diagnostics;

using MachineLearning.NeuralNetwork.Exceptions;

using NeuralNetworks.Layers;
using NeuralNetworks.Operations.Interfaces;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Operations;

public abstract class ParamOperation2D : Operation2D, IParameterCountProvider, IParameterUpdater
{
    public abstract int GetParamCount();
    public abstract void UpdateParams(Layer? layer, Optimizer optimizer);
}

/// <summary>
/// An Operation with parameters of type TParam.
/// </summary>
/// <param name="param">Parameter matrix.</param>
public abstract class ParamOperation2D<TParam>(TParam param) : ParamOperation2D
{
    private TParam? _paramGradient;

    protected TParam Param => param;

    protected TParam ParamGradient => _paramGradient ?? throw new NotYetCalculatedException();

    public override float[,] Backward(float[,] outputGradient)
    {
        float[,] inputGrad = base.Backward(outputGradient);

        _paramGradient = CalcParamGradient(outputGradient);
        EnsureSameShapeForParam(param, _paramGradient);
        return inputGrad;
    }

    [Conditional("DEBUG")]
    protected abstract void EnsureSameShapeForParam(TParam? param, TParam paramGradient);

    protected abstract TParam CalcParamGradient(float[,] outputGradient);

    protected override Operation<float[,], float[,]> CloneBase()
    {
        ParamOperation2D<TParam> clone = (ParamOperation2D<TParam>)base.CloneBase();
        //clone._paramGradient = _paramGradient?.Clone();
        return clone;
    }

}
