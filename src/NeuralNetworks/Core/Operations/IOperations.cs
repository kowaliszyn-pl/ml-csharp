// Neural Networks in C♯
// File name: IOperations.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core.Extensions;

namespace NeuralNetworks.Core.Operations;

public interface IOperations
{
    #region Backend Management

    OperationBackendType BackendType { get; }

    #endregion

    #region Loss Functions

    public float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f);
    public float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target);

    #endregion

    #region Activations Functions

    public float[,,,] LeakyReLU(float[,,,] input, float alpha = 0.01f, float beta = 1f);
    public float[,,,] LeakyReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta);

    public float[,] Tanh(float[,] source);

    /// <summary>
    /// Calculates the gradient of the loss with respect to the input of the Tanh activation function.
    /// </summary>
    /// <remarks>
    /// Given the output gradient (dL/dy), the function calculates the source gradient (dL/dx). 
    /// <para/>
    /// The derivative of the Tanh function <c>tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))</c> is <c>1 - tanh(x)^2</c>.
    /// Therefore, the source gradient is computed as: <c>dL/dx = dL/dy * (1 - tanh(x)^2) = dL/dy * (1 - output^2)</c>.
    /// <list type="bullet">
    /// <item>
    /// tanh(x) => output
    /// </item>
    /// <item>
    /// dL/dy => outputGradient
    /// </item>
    /// <item>
    /// dL/dx => inputGradient
    /// </item>
    /// </list>
    /// </remarks>
    /// <param name="output">The output of the Tanh function (<c>tanh(x)</c>).</param>
    /// <param name="outputGradient">The gradient of the loss with respect to the output of the Tanh function (dL/dy).</param>
    /// <returns>
    /// The gradient of the loss with respect to the input of the Tanh function (dL/dx), having the same shape as <paramref name="outputGradient"/>.
    /// </returns>
    public float[,] TanhInputGradient(float[,] outputGradient, float[,] output);

    public float[,,,] Tanh(float[,,,] source);

    /// <summary>
    /// Calculates the gradient of the loss with respect to the input of the Tanh activation function.
    /// </summary>
    /// <remarks>
    /// Given the output gradient (dL/dy), the function calculates the source gradient (dL/dx). 
    /// <para/>
    /// The derivative of the Tanh function <c>tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))</c> is <c>1 - tanh(x)^2</c>.
    /// Therefore, the source gradient is computed as: <c>dL/dx = dL/dy * (1 - tanh(x)^2) = dL/dy * (1 - output^2)</c>.
    /// <list type="bullet">
    /// <item>
    /// tanh(x) => output
    /// </item>
    /// <item>
    /// dL/dy => outputGradient
    /// </item>
    /// <item>
    /// dL/dx => inputGradient
    /// </item>
    /// </list>
    /// </remarks>
    /// <param name="output">The output of the Tanh function (<c>tanh(x)</c>).</param>
    /// <param name="outputGradient">The gradient of the loss with respect to the output of the Tanh function (dL/dy).</param>
    /// <returns>
    /// The gradient of the loss with respect to the input of the Tanh function (dL/dx), having the same shape as <paramref name="outputGradient"/>.
    /// </returns>
    public float[,,,] TanhInputGradient(float[,,,] outputGradient, float[,,,] output);

    #endregion

    #region Parametric Operations

    // Bias Addition Operations

    public float[,] BiasAddOutput(float[,] input, float[] bias);

    public float[] BiasAddParamGradient(float[,] outputGradient);

    // Convolution Operations

    public float[,,,] Convolve2DOutput(float[,,,] input, float[,,,] weights);
    public float[,,,] Convolve2DInputGradient(float[,,,] input, float[,,,] weights, float[,,,] outputGradient);
    public float[,,,] Convolve2DParamGradient(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth);

    // Weight Multiplication Operations

    public float[,] WeightMultiplyOutput(float[,] input, float[,] weights);
    public float[,] WeightMultiplyInputGradient(float[,] outputGradient, float[,] weights);
    public float[,] WeightMultiplyParamGradient(float[,] input, float[,] outputGradient);

    #endregion

    #region Dropout

    public float[,] DropoutOutput(float[,] input, bool inference, float keepProb, SeededRandom? random, out float[,]? mask);
    public float[,] DropoutInputGradient(float[,] outputGradient, float[,] mask);
    public float[,,,] DropoutOutput(float[,,,] input, bool inference, float keepProb, SeededRandom? random, out float[,,,]? mask);
    public float[,,,] DropoutInputGradient(float[,,,] outputGradient, float[,,,] mask);
    public float[,] InvertedDropoutOutput(float[,] input, bool inference, float keepProb, SeededRandom? random, out float[,]? mask);
    public float[,] InvertedDropoutInputGradient(float[,] outputGradient, float[,] mask, float keepProb);

    #endregion

    #region Transformations

    public float[,] Flatten(float[,,,] source);
    public float[,,,] Unflatten(float[,] source, float[,,,] targetSize);

    #endregion

}