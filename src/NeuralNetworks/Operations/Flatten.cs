// Machine Learning Utils
// File name: Flatten.cs
// Code It Yourself with .NET, 2024

using static NeuralNetworks.Core.ArrayUtils;

namespace NeuralNetworks.Operations;

public class Flatten : Operation<float[,,,], float[,]>
{
    protected override float[,,,] CalcInputGradient(float[,] outputGradient)
    {
        int batchSize = Input.GetLength(0);
        int channels = Input.GetLength(1);
        int height = Input.GetLength(2);
        int width = Input.GetLength(3);

        float[,,,] inputGrad = new float[batchSize, channels, height, width];

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

        return inputGrad;
    }

    protected override float[,] CalcOutput(bool inference)
    {
        // Flattent the input for each batch

        int batchSize = Input.GetLength(0);
        int channels = Input.GetLength(1);
        int height = Input.GetLength(2);
        int width = Input.GetLength(3);

        float[,] output = new float[batchSize, channels * height * width];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int index = c * height * width + h * width + w;
                        output[b, index] = Input[b, c, h, w];
                    }
                }
            }
        }

        return output;
    }

    protected override void EnsureSameShapeForInput(float[,,,]? input, float[,,,] inputGradient)
        => EnsureSameShape(input, inputGradient);

    protected override void EnsureSameShapeForOutput(float[,]? output, float[,] outputGradient)
        => EnsureSameShape(output, outputGradient);
}
