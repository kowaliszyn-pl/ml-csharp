// Machine Learning Utils
// File name: Optimizer.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.


using MachineLearning.NeuralNetwork.LearningRates;

using NeuralNetworks.Layers;

namespace NeuralNetworks.Optimizers;

/// <summary>
/// Base class for a neural network optimizer.
/// </summary>
public abstract class Optimizer(LearningRate learningRate)
{
    protected LearningRate LearningRate => learningRate;

    //public abstract void Step(NeuralNetwork neuralNetwork);

    public virtual void UpdateLearningRate(int epoch, int epochs) => learningRate.Update(epoch, epochs);

    // TODO:: I'll be working to deduplicate these Update procedures somehow

    public abstract void Update(Layer? layer, float[] param, float[] paramGradient);

    public abstract void Update(Layer? layer, float[,] param, float[,] paramGradient);

    public abstract void Update(Layer? layer, float[,,,] param, float[,,,] paramGradient);
}
