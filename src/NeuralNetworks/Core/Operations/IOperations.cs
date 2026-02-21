// Neural Networks in C♯
// File name: IOperations.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Core.Operations;

public interface IOperations
{
    #region Backend Management

    public OperationBackendType BackendType { get; }

    #endregion

    #region Loss Functions

    public float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f);
    public float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target);

    float BinaryCrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f);
    float[,] BinaryCrossEntropyLossGradient(float[,] predicted, float[,] target);

    #endregion

    #region Activations Functions

    public float[,] BipolarSigmoidOutput(float[,] input, float scale);
    public float[,] BipolarSigmoidInputGradient(float[,] outputGradient, float[,] output, float scale);

    public float[,,,] LeakyReLUOutput(float[,,,] input, float alpha = 0.01f, float beta = 1f);
    public float[,,,] LeakyReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta);

    public float[,,] LeakyReLUOutput(float[,,] input, float alpha = 0.01f, float beta = 1f);
    public float[,,] LeakyReLUInputGradient(float[,,] outputGradient, float[,,] input, float alfa, float beta);

    public float[,] LeakyReLUOutput(float[,] input, float alpha = 0.01f, float beta = 1f);
    public float[,] LeakyReLUInputGradient(float[,] outputGradient, float[,] input, float alfa, float beta);

    /// <summary>
    /// Applies the rectified linear unit (ReLU) activation function to each element of the specified 2D array.
    /// </summary>
    /// <remarks>The ReLU function sets all negative values to zero and multiplies non-negative values by the
    /// specified beta. The original array is not modified.</remarks>
    /// <param name="input">The two-dimensional array of single-precision floating-point values to which the ReLU function is applied.</param>
    /// <param name="beta">An optional scaling factor applied to non-negative values. The default is 1.0.</param>
    /// <returns>A new two-dimensional array where each element is the result of applying the ReLU function to the corresponding
    /// element in the source array.</returns>
    public float[,] ReLUOutput(float[,] input, float beta = 1f);
    public float[,] ReLUInputGradient(float[,] outputGradient, float[,] input, float beta);

    public float[,,] ReLUOutput(float[,,] input, float beta = 1f);
    public float[,,] ReLUInputGradient(float[,,] outputGradient, float[,,] input, float beta);

    public float[,,,] ReLUOutput(float[,,,] input, float beta = 1f);
    public float[,,,] ReLUInputGradient(float[,,,] outputGradient, float[,,,] input, float beta);

    public float[,] SigmoidOutput(float[,] input);
    public float[,] SigmoidInputGradient(float[,] outputGradient, float[,] output);

    public float[,] SoftplusOutput(float[,] input);
    public float[,] SoftplusInputGradient(float[,] outputGradient, float[,] output);

    public float[,] TanhOutput(float[,] source);

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

    public float[,,,] TanhOutput(float[,,,] source);

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
    public float[,,] BiasAddConv1DOutput(float[,,] input, float[] bias);
    public float[] BiasAddConv1DParamGradient(float[,,] outputGradient);
    public float[,,,] BiasAddConv2DOutput(float[,,,] input, float[] bias);
    public float[] BiasAddConv2DParamGradient(float[,,,] outputGradient);

    // Convolution Operations

    public float[,,] Convolve1DOutput(float[,,] input, float[,,] weights, int padding, int stride = 1, int dilatation = 1);
    public float[,,] Convolve1DInputGradient(float[,,] input, float[,,] weights, float[,,] outputGradient, int padding, int stride = 1, int dilatation = 1);
    public float[,,] Convolve1DParamGradient(float[,,] input, float[,,] outputGradient, int kernelLength, int padding, int stride = 1, int dilatation = 1);
    public float[,,,] Convolve2DOutput(float[,,,] input, float[,,,] weights, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 1, int dilatationWidth = 1);
    public float[,,,] Convolve2DInputGradient(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 1, int dilatationWidth = 1);
    public float[,,,] Convolve2DParamGradient(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int paddingHeight, int paddingWidth, int strideHeight = 1, int strideWidth = 1, int dilatationHeight = 1, int dilatationWidth = 1);

    // Weight Multiplication Operations

    public float[,] WeightMultiplyOutput(float[,] input, float[,] weights);
    public float[,] WeightMultiplyInputGradient(float[,] outputGradient, float[,] weights);
    public float[,] WeightMultiplyParamGradient(float[,] input, float[,] outputGradient);

    #endregion

    #region Dropout

    public float[,] DropoutOutput(float[,] input, bool inference, float keepProb, SeededRandom? random, out float[,]? mask);
    public float[,] DropoutInputGradient(float[,] outputGradient, float[,] mask);

    public float[,,] DropoutOutput(float[,,] input, bool inference, float keepProb, SeededRandom? random, out float[,,]? mask);
    public float[,,] DropoutInputGradient(float[,,] outputGradient, float[,,] mask);

    public float[,,,] DropoutOutput(float[,,,] input, bool inference, float keepProb, SeededRandom? random, out float[,,,]? mask);
    public float[,,,] DropoutInputGradient(float[,,,] outputGradient, float[,,,] mask);

    public float[,] InvertedDropoutOutput(float[,] input, bool inference, float keepProb, SeededRandom? random, out float[,]? mask);
    public float[,] InvertedDropoutInputGradient(float[,] outputGradient, float[,] mask, float keepProb);

    #endregion

    #region Transformations

    public float[,] Flatten(float[,,,] source);
    public float[,,,] Unflatten(float[,] source, float[,,,] targetSize);
    
    float[,,] MaxPooling1DOutput(float[,,] input, int size, out int[,,] maxIndices);
    float[,,] MaxPooling1DInputGradient(float[,,] input, float[,,] outputGradient, int size, int[,,] maxIndices);

    float[,] GlobalAveragePooling1DOutput(float[,,] input);
    float[,,] GlobalAveragePooling1DInputGradient(float[,,] input, float[,] outputGradient);

    #endregion

}