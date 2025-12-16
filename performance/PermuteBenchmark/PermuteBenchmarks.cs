// Neural Networks in C♯
// File name: PermuteBenchmarks.cs
// www.kowaliszyn.pl, 2025

using System;

using BenchmarkDotNet.Attributes;

using Microsoft.VSDiagnostics;

using NeuralNetworks.Core;

namespace PermuteBenchmark;
// For more information on the VS BenchmarkDotNet Diagnosers see https://learn.microsoft.com/visualstudio/profiling/profiling-with-benchmark-dotnet
[CPUUsageDiagnoser]
public class PermuteBenchmarks
{
    public PermuteBenchmarks()
    {
        CreateX4();
        CreateX2();
        CreateY();
    }

    private const int Rows = 10000;
    private float[,,,] _x4;
    private float[,] _x2;
    private float[,] _y;

    private void CreateX4()
    {
        _x4 = new float[Rows, 100, 5, 5];
        SeededRandom random = new(251207);
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                // _x4[i, j] = random.NextSingle();
                for (int k = 0; k < 5; k++)
                {
                    for (int l = 0; l < 5; l++)
                    {
                        _x4[i, j, k, l] = random.NextSingle();
                    }
                }
            }
        }
    }

    private void CreateX2()
    {
        _x2 = new float[Rows, 100];
        SeededRandom random = new(251207);
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                _x2[i, j] = random.NextSingle();
            }
        }
    }

    private void CreateY()
    {
        _y = new float[Rows, 1];
        SeededRandom random = new(251207);
        for (int i = 0; i < Rows; i++)
        {
            _y[i, 0] = random.NextSingle();
        }
    }

    [Benchmark]
    public void PermuteData4()
    {
        Random random = new SeededRandom(251207);
        (float[,,,] newX, float[,] newY) = ArrayUtils.PermuteData(_x4, _y, random);
    }

    [Benchmark]
    public void PermuteData2()
    {
        Random random = new SeededRandom(251207);
        (float[,] newX, float[,] newY) = ArrayUtils.PermuteData(_x2, _y, random);
    }

    [Benchmark]
    public void PermuteInPlaceTogetherWith4()
    {
        Random random = new SeededRandom(251207);
        _x4.PermuteInPlaceTogetherWith(_y, random);
    }

    [Benchmark]
    public void PermuteInPlaceTogetherWith2()
    {
        Random random = new SeededRandom(251207);
        _x2.PermuteInPlaceTogetherWith(_y, random);
    }

    [Benchmark]
    public void PermuteInPlaceTogetherWithSetRow4()
    {
        Random random = new SeededRandom(251207);
        _x4.PermuteInPlaceTogetherWithSetRow(_y, random);
    }

    [Benchmark]
    public void PermuteInPlaceTogetherWithSetRow2()
    {
        Random random = new SeededRandom(251207);
        _x2.PermuteInPlaceTogetherWithSetRow(_y, random);
    }

}
