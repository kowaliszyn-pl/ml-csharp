// Neural Networks in C♯
// File name: Optimizer.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Layers;
using NeuralNetworks.LearningRates;

namespace NeuralNetworks.Optimizers;

/// <summary>
/// Base class for a neural network optimizer.
/// </summary>
public abstract class Optimizer(LearningRate learningRate)
{
    public LearningRate LearningRate => learningRate;

    //public abstract void Step(GenericModel neuralNetwork);

    public virtual void UpdateLearningRate(int epoch, int epochs) => learningRate.Update(epoch, epochs);

    // TODO:: I'll be working to deduplicate these Update procedures somehow

    public abstract void Update(Layer? layer, float[] param, float[] paramGradient);

    public abstract void Update(Layer? layer, float[,] param, float[,] paramGradient);

    public abstract void Update(Layer? layer, float[,,,] param, float[,,,] paramGradient);
}
