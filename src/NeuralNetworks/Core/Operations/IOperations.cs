// Neural Networks in C♯
// File name: IOperations.cs
// www.kowaliszyn.pl, 2025

namespace NeuralNetworks.Core.Operations;

internal interface IOperations
{
    public float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f);
    public float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target);
    public float[,] WeightMultiplyCalcOutput(float[,] input, float[,] weights);
    public float[,] WeightMultiplyCalcInputGradient(float[,] outputGradient, float[,] weights);
    public float[,] WeightMultiplyCalcParamGradient(float[,] input, float[,] outputGradient)
}
