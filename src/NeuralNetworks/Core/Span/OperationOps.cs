// Neural Networks in C♯
// File name: OperationOps.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NeuralNetworks.Core.Span;

public static class OperationOps
{
    /// <summary>
    /// 2D convolution forward pass on NHWC-like 4D tensors:
    /// Input: [batch, inChannels, inHeight, inWidth]
    /// Weights: [inChannels, outChannels, kernelHeight, kernelWidth]
    /// Output: [batch, outChannels, outHeight, outWidth]
    /// Padding is symmetric and computed as kernelSize / 2 (same padding) if not specified.
    /// </summary>
    /// <returns>Output</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Convolve2DForward(float[,,,] input, float[,,,] weights, int? padding = null)
    {
        int batchSize = input.GetLength(0);

        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int weightChannels = weights.GetLength(0);
        int outputChannels = weights.GetLength(1);
        int kernelHeight = weights.GetLength(2);
        int kernelWidth = weights.GetLength(3);

        Debug.Assert(weightChannels == inputChannels);
        Debug.Assert(kernelHeight == kernelWidth);

        int pad = padding ?? (kernelHeight / 2);

        int outputHeight = inputHeight - kernelHeight + 1 + 2 * pad;
        int outputWidth = inputWidth - kernelWidth + 1 + 2 * pad;

        float[,,,] output = new float[batchSize, outputChannels, outputHeight, outputWidth];

        ref float inputRef = ref input[0, 0, 0, 0];
        ref float weightsRef = ref weights[0, 0, 0, 0];
        ref float outputRef = ref output[0, 0, 0, 0];

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref inputRef, input.Length);
        ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weightsRef, weights.Length);
        Span<float> outputSpan = MemoryMarshal.CreateSpan(ref outputRef, output.Length);

        // pre-compute sizes for offsets
        int outputBSize = outputChannels * outputHeight * outputWidth;
        int outputCSize = outputHeight * outputWidth;
        int inputBSize = inputChannels * inputHeight * inputWidth;
        int inputCSize = inputHeight * inputWidth;
        int weightsCSize = outputChannels * kernelHeight * kernelWidth;
        int weightsOutputCSize = kernelHeight * kernelWidth;

        for (int b = 0; b < batchSize; b++)
        {
            int inputBIndex = b * inputBSize;
            int outputBIndex = b * outputBSize;
            for (int oc = 0; oc < outputChannels; oc++)
            {
                int weightsOutputCIndex = oc * weightsOutputCSize;
                int outputCIndex = oc * outputCSize;
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    int outputHIndex = oh * outputWidth;
                    int ohMinusPad = oh - pad;
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int owMinusPad = ow - pad;
                        float sum = 0f;
                        for (int ic = 0; ic < inputChannels; ic++)
                        {
                            int inputCIndex = ic * inputCSize;
                            int weightsInputCIndex = ic * weightsCSize;
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                int weightsKernelHIndex = kh * kernelWidth;
                                int ih = kh + ohMinusPad;
                                if (ih >= 0 && ih < inputHeight)
                                {
                                    int inputHIndex = ih * inputWidth;
                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {
                                        int iw = kw + owMinusPad;
                                        if (iw >= 0 && iw < inputWidth)
                                        {
                                            // sum += input[b, ic, ih, iw] * weights[ic, oc, kh, kw];
                                            sum += inputSpan[inputBIndex + inputCIndex + inputHIndex + iw] *
                                                   weightsSpan[weightsInputCIndex + weightsOutputCIndex + weightsKernelHIndex + kw];
                                        }
                                    }
                                }
                            }
                        }
                        // output[b, oc, oh, ow] = sum;
                        outputSpan[outputBIndex + outputCIndex + outputHIndex + ow] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Backward pass w.r.t. input for 2D convolution.
    /// inputGrad shape: [batch, inChannels, inHeight, inWidth]
    /// outputGrad shape: [batch, outChannels, outHeight, outWidth]
    /// weights shape: [inChannels, outChannels, kernelHeight, kernelWidth]
    /// </summary>
    /// <returns>InputGradient</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Convolve2DBackwardInput(float[,,,] input, float[,,,] weights, float[,,,] outputGradient, int? padding = null)
    {
        int batchSize = outputGradient.GetLength(0);

        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputGradientChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        int kernelHeight = weights.GetLength(2);
        int kernelWidth = weights.GetLength(3);

        Debug.Assert(weights.GetLength(0) == inputChannels);
        Debug.Assert(weights.GetLength(1) == outputGradientChannels);
        Debug.Assert(kernelHeight == kernelWidth);

        int pad = padding ?? (kernelHeight / 2);

        float[,,,] inputGradient = new float[batchSize, inputChannels, inputHeight, inputWidth];

        ref float weightsRef = ref weights[0, 0, 0, 0];
        ref float outputGradientRef = ref outputGradient[0, 0, 0, 0];
        ref float inputGradientRef = ref inputGradient[0, 0, 0, 0];

        ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weightsRef, weights.Length);
        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradientRef, outputGradient.Length);
        Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradientRef, inputGradient.Length);

        // pre-compute sizes for offsets
        int outputGradientBSize = outputGradientChannels * outputGradientHeight * outputGradientWidth;
        int outputGradientCSize = outputGradientHeight * outputGradientWidth;
        int weightsInputCSize = outputGradientChannels * kernelHeight * kernelWidth;
        int weightsOutputCSize = kernelHeight * kernelWidth;
        int inputGradientBSize = inputChannels * inputHeight * inputWidth;
        int inputGradientCSize = inputHeight * inputWidth;

        for (int b = 0; b < batchSize; b++)
        {
            int outputGradientBIndex = b * outputGradientBSize;
            for (int ic = 0; ic < inputChannels; ic++)
            {
                int weightsInputCIndex = ic * weightsInputCSize;
                int inputGradientCIndex = ic * inputGradientCSize;
                for (int ih = 0; ih < inputHeight; ih++)
                {
                    int inputGradientHIndex = ih * inputWidth;
                    int ihPlusPad = ih + pad;
                    for (int iw = 0; iw < inputWidth; iw++)
                    {
                        float sum = 0f;
                        int inputGradientBIndex = b * inputGradientBSize;
                        int iwPlusPad = iw + pad;
                        for (int oc = 0; oc < outputGradientChannels; oc++)
                        {
                            int outputGradientCIndex = oc * outputGradientCSize;
                            int weightsOutputCIndex = oc * weightsOutputCSize;
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                int oh = ihPlusPad - kh;
                                if (oh >= 0 && oh < outputGradientHeight)
                                {
                                    int weightsKernelHIndex = kh * kernelWidth;
                                    int outputHIndex = oh * outputGradientWidth;
                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {
                                        int ow = iwPlusPad - kw;
                                        if (ow >= 0 && ow < outputGradientWidth)
                                        {
                                            // sum += outputGradient[b, oc, oh, ow] * weights[ic, oc, kh, kw];
                                            sum += outputGradientSpan[outputGradientBIndex + outputGradientCIndex + outputHIndex + ow]
                                                * weightsSpan[weightsInputCIndex + weightsOutputCIndex + weightsKernelHIndex + kw];
                                        }
                                    }
                                }
                            }
                        }
                        // inputGradient[b, ic, ih, iw] = sum;
                        inputGradientSpan[inputGradientBIndex + inputGradientCIndex + inputGradientHIndex + iw] = sum;
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Backward pass w.r.t. weights (parameters) for 2D convolution.
    /// Returns gradient with shape [inChannels, outChannels, kernelHeight, kernelWidth].
    /// </summary>
    /// <returns>ParamGradient</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Convolve2DBackwardWeights(float[,,,] input, float[,,,] outputGradient, int kernelHeight, int kernelWidth, int? padding = null)
    {
        int batchSize = outputGradient.GetLength(0);

        int inputChannels = input.GetLength(1);
        int inputHeight = input.GetLength(2);
        int inputWidth = input.GetLength(3);

        int outputGradientChannels = outputGradient.GetLength(1);
        int outputGradientHeight = outputGradient.GetLength(2);
        int outputGradientWidth = outputGradient.GetLength(3);

        Debug.Assert(kernelHeight == kernelWidth);
        int pad = padding ?? (kernelHeight / 2);

        float[,,,] paramGradient = new float[inputChannels, outputGradientChannels, kernelHeight, kernelWidth];

        ref float inputRef = ref input[0, 0, 0, 0];
        ref float outputGradientRef = ref outputGradient[0, 0, 0, 0];

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref inputRef, input.Length);
        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradientRef, outputGradient.Length);
        Span<float> paramGradientSpan = MemoryMarshal.CreateSpan(ref paramGradient[0, 0, 0, 0], paramGradient.Length);

        // pre-compute sizes for offsets
        int outputGradientBSize = outputGradientChannels * outputGradientHeight * outputGradientWidth;
        int outputGradientOutputCSize = outputGradientHeight * outputGradientWidth;
        int inputBSize = inputChannels * inputHeight * inputWidth;
        int inputCSize = inputHeight * inputWidth;
        int paramGradientInputCSize = outputGradientChannels * kernelHeight * kernelWidth;
        int paramGradientOutputCSize = kernelHeight * kernelWidth;

        for (int b = 0; b < batchSize; b++)
        {
            int outputGradientBIndex = b * outputGradientBSize;
            int inputBIndex = b * inputBSize;
            for (int ic = 0; ic < inputChannels; ic++)
            {
                int inputCIndex = ic * inputCSize;
                int paramGradientInputCIndex = ic * paramGradientInputCSize;
                for (int oc = 0; oc < outputGradientChannels; oc++)
                {
                    int outputGradientOutputCIndex = oc * outputGradientOutputCSize;
                    int paramGradientOutputCIndex = oc * paramGradientOutputCSize;
                    for (int kh = 0; kh < kernelHeight; kh++)
                    {
                        int paramGradientKernelHIndex = kh * kernelWidth;
                        int khMinusPad = kh - pad;
                        for (int kw = 0; kw < kernelWidth; kw++)
                        {
                            int kwMinusPad = kw - pad;
                            float sum = 0f;
                            for (int oh = 0; oh < outputGradientHeight; oh++)
                            {
                                int ih = oh + khMinusPad;
                                if (ih >= 0 && ih < inputHeight)
                                {
                                    int inputHIndex = ih * inputWidth;
                                    int outputGradientHIndex = oh * outputGradientWidth;
                                    for (int ow = 0; ow < outputGradientWidth; ow++)
                                    {
                                        int iw = ow + kwMinusPad;
                                        if (iw >= 0 && iw < inputWidth)
                                        {
                                            // sum += outputGradient[b, oc, oh, ow] * input[b, ic, ih, iw]
                                            sum += outputGradientSpan[outputGradientBIndex + outputGradientOutputCIndex + outputGradientHIndex + ow] *
                                                   inputSpan[inputBIndex + inputCIndex + inputHIndex + iw];
                                        }
                                    }
                                }
                            }
                            // paramGradient[ic, oc, kh, kw] += sum;
                            paramGradientSpan[paramGradientInputCIndex + paramGradientOutputCIndex + paramGradientKernelHIndex + kw] += sum;
                        }
                    }
                }
            }
        }

        return paramGradient;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float CrossEntropyLoss(float[,] predicted, float[,] target, float eps = 1e-7f)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        float loss = 0f;
        int batchSize = predicted.GetLength(0);
        int numClasses = predicted.GetLength(1);

        ref float predictedRef = ref predicted[0, 0];
        ref float targetRef = ref target[0, 0];

        ReadOnlySpan<float> predictedSpan = MemoryMarshal.CreateReadOnlySpan(ref predictedRef, predicted.Length);
        ReadOnlySpan<float> targetSpan = MemoryMarshal.CreateReadOnlySpan(ref targetRef, target.Length);

        for (int i = 0; i < targetSpan.Length; i++)
        {
            float p = Math.Clamp(predictedSpan[i], eps, 1 - eps);
            loss += targetSpan[i] * MathF.Log(p);
        }

        return -loss / (batchSize * numClasses);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] CrossEntropyLossGradient(float[,] predicted, float[,] target)
    {
        Debug.Assert(predicted.Length == target.Length, "Predicted and target arrays must have the same length.");

        int batchSize = predicted.GetLength(0);
        int numClasses = predicted.GetLength(1);
        float[,] gradient = new float[batchSize, numClasses];

        ref float predictedRef = ref predicted[0, 0];
        ref float targetRef = ref target[0, 0];
        ref float gradientRef = ref gradient[0, 0];

        ReadOnlySpan<float> predictedSpan = MemoryMarshal.CreateReadOnlySpan(ref predictedRef, predicted.Length);
        ReadOnlySpan<float> targetSpan = MemoryMarshal.CreateReadOnlySpan(ref targetRef, target.Length);
        Span<float> gradientSpan = MemoryMarshal.CreateSpan(ref gradientRef, gradient.Length);

        for (int i = 0; i < gradientSpan.Length; i++)
        {
            gradientSpan[i] = (predictedSpan[i] - targetSpan[i]) / batchSize;
        }

        return gradient;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] LeakyReLUCalcInputGradient(float[,,,] outputGradient, float[,,,] input, float alfa, float beta)
    {
        int dim0 = outputGradient.GetLength(0);
        int dim1 = outputGradient.GetLength(1);
        int dim2 = outputGradient.GetLength(2);
        int dim3 = outputGradient.GetLength(3);

        Debug.Assert(dim0 > 0 && dim1 > 0 && dim2 > 0 && dim3 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(input.GetLength(0) == dim0 && input.GetLength(1) == dim1 && input.GetLength(2) == dim2 && input.GetLength(3) == dim3, "Shapes of outputGradient and input must match for elementwise operations.");

        float[,,,] result = new float[dim0, dim1, dim2, dim3];
        ref float outputGradientRef = ref outputGradient[0, 0, 0, 0];
        ref float inputRef = ref input[0, 0, 0, 0];
        ref float resultRef = ref result[0, 0, 0, 0];

        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradientRef, outputGradient.Length);
        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref inputRef, input.Length);
        Span<float> resultSpan = MemoryMarshal.CreateSpan(ref resultRef, result.Length);

        for (int i = 0; i < resultSpan.Length; i++)
        {
            resultSpan[i] = inputSpan[i] > 0 ? outputGradientSpan[i] * beta : outputGradientSpan[i] * alfa;
        }
        return result;
    }

    /// <param name="outputGradient"></param>
    /// <param name="output"></param>
    /// <returns>InputGradient</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] MultiplyByTanhDerivative(float[,,,] outputGradient, float[,,,] output)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Tanh function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Tanh function tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) is 1 - tanh(x)^2.
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * (1 - tanh(x)^2).
        // The elementwise multiplication of the output gradient and the derivative of the Tanh function is returned as the input gradient.
        // tanh(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient

        int dim0 = outputGradient.GetLength(0);
        int dim1 = outputGradient.GetLength(1);
        int dim2 = outputGradient.GetLength(2);
        int dim3 = outputGradient.GetLength(3);

        Debug.Assert(dim0 > 0 && dim1 > 0 && dim2 > 0 && dim3 > 0, "All dimensions must be greater than zero.");
        Debug.Assert(output.GetLength(0) == dim0 && output.GetLength(1) == dim1 && output.GetLength(2) == dim2 && output.GetLength(3) == dim3, "Shapes of outputGradient and output must match for elementwise operations.");

        float[,,,] result = new float[dim0, dim1, dim2, dim3];

        ref float ogRef = ref outputGradient[0, 0, 0, 0];
        ref float outRef = ref output[0, 0, 0, 0];
        ref float resRef = ref result[0, 0, 0, 0];

        ReadOnlySpan<float> ogSpan = MemoryMarshal.CreateReadOnlySpan(ref ogRef, outputGradient.Length);
        ReadOnlySpan<float> outSpan = MemoryMarshal.CreateReadOnlySpan(ref outRef, output.Length);
        Span<float> resSpan = MemoryMarshal.CreateSpan(ref resRef, result.Length);

        for (int i = 0; i < resSpan.Length; i++)
        {
            float y = outSpan[i];
            resSpan[i] = ogSpan[i] * (1f - (y * y));
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] WeightMultiplyCalcOutput(float[,] input, float[,] weights)
    {
        int batchSize = input.GetLength(0);
        int inputFeatures = input.GetLength(1);
        int outputFeatures = weights.GetLength(1);

        Debug.Assert(weights.GetLength(0) == inputFeatures, "Input features must match weight input dimension.");

        float[,] output = new float[batchSize, outputFeatures];

        ref float inputRef = ref input[0, 0];
        ref float weightsRef = ref weights[0, 0];
        ref float outputRef = ref output[0, 0];

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref inputRef, input.Length);
        ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weightsRef, weights.Length);
        Span<float> outputSpan = MemoryMarshal.CreateSpan(ref outputRef, output.Length);

        for (int b = 0; b < batchSize; b++)
        {
            int inputBIndex = b * inputFeatures;
            int outputBIndex = b * outputFeatures;
            for (int ofeature = 0; ofeature < outputFeatures; ofeature++)
            {
                float sum = 0f;
                for (int ifeature = 0; ifeature < inputFeatures; ifeature++)
                {
                    // sum += input[b, inputFeature] * weights[inputFeature, outputFeature];
                    sum += inputSpan[inputBIndex + ifeature] * weightsSpan[ifeature * outputFeatures + ofeature];
                }
                // output[b, outputFeature] = sum;
                outputSpan[outputBIndex + ofeature] = sum;
            }
        }

        Debug.Assert(output.Length == batchSize * outputFeatures, "Output shape is incorrect.");

        return output;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] WeightMultiplyCalcInputGradient(float[,] outputGradient, float[,] weights)
    {
        // outputGradient.MultiplyDot(weights.Transpose());

        int batchSize = outputGradient.GetLength(0); // 100
        int inputFeatures = weights.GetLength(0); // 46
        int outputFeatures = weights.GetLength(1); // 10
        int outputGradientFeatures = outputGradient.GetLength(1); // 10

        Debug.Assert(outputGradientFeatures == outputFeatures, "Output features of output gradient must match weight output dimension.");

        float[,] inputGradient = new float[batchSize, inputFeatures];
        ref float outputGradientRef = ref outputGradient[0, 0];
        ref float weightsRef = ref weights[0, 0];
        ref float inputGradientRef = ref inputGradient[0, 0];

        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradientRef, outputGradient.Length);
        ReadOnlySpan<float> weightsSpan = MemoryMarshal.CreateReadOnlySpan(ref weightsRef, weights.Length);
        Span<float> inputGradientSpan = MemoryMarshal.CreateSpan(ref inputGradientRef, inputGradient.Length);

        for (int b = 0; b < batchSize; b++)
        {
            int outputGradientBIndex = b * outputFeatures;
            int inputGradientBIndex = b * inputFeatures;
            for (int inputFeature = 0; inputFeature < inputFeatures; inputFeature++)
            {
                float sum = 0f;
                for (int outputFeature = 0; outputFeature < outputFeatures; outputFeature++)
                {
                    // sum += outputGradient[b, outputFeature] * weights[inputFeature, outputFeature];
                    sum += outputGradientSpan[outputGradientBIndex + outputFeature] * weightsSpan[inputFeature * outputFeatures + outputFeature];
                }
                // inputGradient[b, inputFeature] = sum;
                inputGradientSpan[inputGradientBIndex + inputFeature] = sum;
            }
        }

        return inputGradient;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] WeightMultiplyCalcParamGradient(float[,] input, float[,] outputGradient)
    { 
        // input.Transpose().MultiplyDot(outputGradient);

        int batchSize = input.GetLength(0);
        int inputFeatures = input.GetLength(1);
        int outputFeatures = outputGradient.GetLength(1);

        Debug.Assert(outputGradient.GetLength(0) == batchSize, "Batch size of output gradient must match batch size of input.");

        float[,] paramGradient = new float[inputFeatures, outputFeatures];
        ref float inputRef = ref input[0, 0];
        ref float outputGradientRef = ref outputGradient[0, 0];
        ref float paramGradientRef = ref paramGradient[0, 0];

        ReadOnlySpan<float> inputSpan = MemoryMarshal.CreateReadOnlySpan(ref inputRef, input.Length);
        ReadOnlySpan<float> outputGradientSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradientRef, outputGradient.Length);
        Span<float> paramGradientSpan = MemoryMarshal.CreateSpan(ref paramGradientRef, paramGradient.Length);

        for (int ifeature = 0; ifeature < inputFeatures; ifeature++)
        {
            int paramGradientInputFeatureIndex = ifeature * outputFeatures;
            for (int ofeature = 0; ofeature < outputFeatures; ofeature++)
            {
                float sum = 0f;
                for (int b = 0; b < batchSize; b++)
                {
                    // sum += input[b, inputFeature] * outputGradient[b, outputFeature];
                    sum += inputSpan[b * inputFeatures + ifeature] * outputGradientSpan[b * outputFeatures + ofeature];
                }
                // paramGradient[inputFeature, outputFeature] = sum;
                paramGradientSpan[paramGradientInputFeatureIndex + ofeature] = sum;
            }
        }
        return paramGradient;
    }
}
