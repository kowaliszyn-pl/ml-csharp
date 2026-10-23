// Neural Networks in C♯
// File name: DataLoaderRoutines.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core;

public static class DataLoaderRoutines
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] LoadCsv(string filePath, int skipHeaderLines = 0)
       => LoadSv(filePath, ',', skipHeaderLines);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] LoadTsv(string filePath, int skipHeaderLines = 0)
        => LoadSv(filePath, '\t', skipHeaderLines);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float[,] LoadSv(string filePath, char separator, int skipHeaderLines)
    {
        string[] lines = [.. File.ReadAllLines(filePath).Skip(skipHeaderLines)];
        int rows = lines.Length;
        int cols = lines[0].Split(separator).Length;
        float[,] matrix = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            string[] values = lines[i].Split(separator);
            for (int j = 0; j < cols; j++)
            {
                string value = values[j].Trim('"');
                matrix[i, j] = float.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
            }
        }
        return matrix;
    }

}
