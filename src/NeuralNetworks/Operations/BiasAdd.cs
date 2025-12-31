// Neural Networks in C♯
// File name: BiasAdd.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core.Extensions;
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
       // outputGradient is already averaged over the batch size in the loss function, so we just need to sum by columns
       => outputGradient.SumByColumns();

    protected override float[,] CalcInputGradient(float[,] outputGradient)
      => outputGradient; // => Input.AsOnes().MultiplyElementwise(outputGradient);

    public override void UpdateParams(Layer? layer, Optimizer optimizer) 
        => optimizer.Update(layer, Param, ParamGradient);

    protected override void EnsureSameShapeForParam(float[]? param, float[] paramGradient)
        => EnsureSameShape(param, paramGradient);

    public override int GetParamCount()
        => Param.Length;
}
