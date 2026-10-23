// Neural Networks in C♯
// File name: ReshapeUtils.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

public static class ReshapeRoutines
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] ReshapeTo2D(this float[,,,] data)
    {
        int rows = data.GetLength(0);
        int channels = data.GetLength(1);
        int height = data.GetLength(2);
        int width = data.GetLength(3);

        float[,] res = new float[rows, channels * height * width];
        for (int i = 0; i < rows; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                int channelOffset = c * height * width;
                for (int h = 0; h < height; h++)
                {
                    int channelHeightOffset = channelOffset + h * width;
                    for (int w = 0; w < width; w++)
                    {
                        res[i, channelHeightOffset + w] = data[i, c, h, w];
                    }
                }
            }
        }
        return res;
    }

    // TODO: round-trip tests 4D -> 2D -> 4D preserving values for a few non-trivial shapes

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] ReshapeTo4D(this float[,] data, int channels, int height, int width)
    {
        int rows = data.GetLength(0);
        int columns = data.GetLength(1);

        Debug.Assert(columns == channels * height * width);

        float[,,,] res = new float[rows, channels, height, width];
        for (int i = 0; i < rows; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                int channelOffset = c * height * width;
                for (int h = 0; h < height; h++)
                {
                    int channelHeightOffset = channelOffset + h * width;
                    for (int w = 0; w < width; w++)
                    {
                        res[i, c, h, w] = data[i, channelHeightOffset + w];
                    }
                }
            }
        }
        return res;
    }

}
