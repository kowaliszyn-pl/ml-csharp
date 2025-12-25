// Machine Learning Utils
// File name: Flatten.cs
// Code It Yourself with .NET, 2024

using NeuralNetworks.Core.Span;

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations;

public class Flatten : Operation<float[,,,], float[,]>
{
    protected override float[,] CalcOutput(bool inference)
    {
        return Input.Flatten();
    }

    protected override float[,,,] CalcInputGradient(float[,] outputGradient)
    {
        return outputGradient.Unflatten(Input);
        /*
        int batchSize = Input.GetLength(0);
        int channels = Input.GetLength(1);
        int height = Input.GetLength(2);
        int width = Input.GetLength(3);

        float[,,,] inputGrad = new float[batchSize, channels, height, width];
#if SPAN
        ReadOnlySpan<float> outputGradSpan = MemoryMarshal.CreateReadOnlySpan(ref outputGradient[0, 0], outputGradient.Length);
        Span<float> inputGradSpan = MemoryMarshal.CreateSpan(ref inputGrad[0, 0, 0, 0], inputGrad.Length);
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int index = c * height * width + h * width + w;
                        inputGradSpan[b * (channels * height * width) + c * (height * width) + h * width + w] =
                            outputGradSpan[b * (channels * height * width) + index];
                    }
                }
            }
        }
#else
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int index = c * height * width + h * width + w;
                        inputGrad[b, c, h, w] = outputGradient[b, index];
                    }
                }
            }
        }
#endif
        return inputGrad;*/
    }

    protected override void EnsureSameShapeForInput(float[,,,]? input, float[,,,] inputGradient)
        => EnsureSameShape(input, inputGradient);

    protected override void EnsureSameShapeForOutput(float[,]? output, float[,] outputGradient)
        => EnsureSameShape(output, outputGradient);
}
