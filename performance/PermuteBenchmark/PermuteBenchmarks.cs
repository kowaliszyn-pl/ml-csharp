// Neural Networks in C♯
// File name: PermuteBenchmarks.cs
// www.kowaliszyn.pl, 2025

using System;
using System.Security.Cryptography;

using BenchmarkDotNet.Attributes;

using Microsoft.VSDiagnostics;

using NeuralNetworks.Core;

namespace PermuteBenchmark;
// For more information on the VS BenchmarkDotNet Diagnosers see https://learn.microsoft.com/visualstudio/profiling/profiling-with-benchmark-dotnet
[CPUUsageDiagnoser]
public class PermuteBenchmarks
{

    float[,,,] x4;
    float[,] x2;
    float[,] y;

    public void CreateArrays4()
    {
        x4 = new float[10000, 100, 5, 5];
        // x4 = new float[10000, 100];
        y = new float[10000, 10];
        SeededRandom random = new(251207);
        for (int i = 0; i < 10000; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                // x4[i, j] = random.NextSingle();
                for (int k = 0; k < 5; k++)
                {
                    for (int l = 0; l < 5; l++)
                    {
                        x4[i, j, k, l] = random.NextSingle();
                    }
                }
            }
            for (int j = 0; j < 10; j++)
            {
                y[i, j] = random.NextSingle();
            }
        }
    }

    public void CreateArrays2()
    {
        x2 = new float[10000, 100];
        // x4 = new float[10000, 100];
        y = new float[10000, 10];
        SeededRandom random = new(251207);
        for (int i = 0; i < 10000; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                x2[i, j] = random.NextSingle();
                //for (int k = 0; k < 5; k++)
                //{
                //    for (int l = 0; l < 5; l++)
                //    {
                //        x4[i, j, k, l] = random.NextSingle();
                //    }
                //}
            }
            for (int j = 0; j < 10; j++)
            {
                y[i, j] = random.NextSingle();
            }
        }
    }

    [Benchmark]
    public void PermuteOld4()
    {
        CreateArrays4();
        Random random = new SeededRandom(251207);
        (float[,,,] newX, float[,] newY) = ArrayUtils.PermuteData(x4, y, random);
        //(float[,] newX, float[,] newY) = ArrayUtils.PermuteData(x4, y, random);
    }

    [Benchmark]
    public void PermuteNew4()
    {
        CreateArrays4();
        Random random = new SeededRandom(251207);
        x4.PermuteInPlaceTogetherWith(y, random);
    }

    [Benchmark]
    public void PermuteNew4SetRow()
    {
        CreateArrays4();
        Random random = new SeededRandom(251207);
        x4.PermuteInPlaceTogetherWithSetRow(y, random);
    }

    [Benchmark]
    public void PermuteOld2()
    {
        CreateArrays2();
        Random random = new SeededRandom(251207);
        (float[,] newX, float[,] newY) = ArrayUtils.PermuteData(x2, y, random);
        //(float[,] newX, float[,] newY) = ArrayUtils.PermuteData(x4, y, random);
    }

    [Benchmark]
    public void PermuteNew2()
    {
        CreateArrays2();
        Random random = new SeededRandom(251207);
        x2.PermuteInPlaceTogetherWith(y, random);
    }

}
