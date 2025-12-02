// Neural Networks in C♯
// File name: BiasAdd.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations;

/// <summary>
/// Computes bias addition.
/// </summary>
/// <param name="bias">Bias matrix.</param>
public class BiasAdd(float[] bias) : ParamOperation2D<float[]>(bias)
{

    protected override float[,] CalcOutput(bool inference)
        => Input.AddRow(Param);

    protected override float[] CalcParamGradient(float[,] outputGradient)
    {
        float[,] paramGrad = Param.AsOnes().MultiplyElementwise(outputGradient);
        // We dont use AvgByRows here, because the outputGradient is already averaged over the batch size in the loss function.
        return paramGrad.SumByColumns();
        // return outputGradient.SumByColumns(); ?
    }

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => Input.AsOnes().MultiplyElementwise(outputGradient);
    //  => outputGradient ?

    public override void UpdateParams(Layer? layer, Optimizer optimizer)
    {
        optimizer.Update(layer, Param, ParamGradient);
    }

    protected override void EnsureSameShapeForParam(float[]? param, float[] paramGradient)
        => EnsureSameShape(param, paramGradient);

    public override int GetParamCount()
        => Param.Length;
}
