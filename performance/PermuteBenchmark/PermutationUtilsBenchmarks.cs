// Neural Networks in C♯
// File name: PermutationUtilsBenchmarks.cs
// www.kowaliszyn.pl, 2025

using System;

using BenchmarkDotNet.Attributes;

using Microsoft.VSDiagnostics;

using NeuralNetworks.Core;
using NeuralNetworks.Core.Utils;

namespace PermuteBenchmark;

[CPUUsageDiagnoser]
public class PermutationUtilsBenchmarks
{
    private const int Rows = 10000;
    private float[,] _x2;
    private float[,] _y;

    public PermutationUtilsBenchmarks()
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

        _y = new float[Rows, 1];
        SeededRandom randomY = new(251207);
        for (int i = 0; i < Rows; i++)
        {
            _y[i, 0] = randomY.NextSingle();
        }
    }

    /*
    // (1) Redundant Array.Clone() in Permute(x,y) overload vs internal clone only (PermuteInPlaceTogetherWith)
    [Benchmark]
    public void Permute_XY_RedundantClone()
    {
        Random random = new SeededRandom(251207);
        (float[,] xRes, float[,] yRes) = PermutationUtils.Permute(_x2, _y, random, PermutationUtils.PermuteMethod.Indices, PermutationUtils.CopyMethod.CopyValues);
    }

    [Benchmark]
    public void PermuteInPlaceTogetherWith_SingleClone()
    {
        float[,] xCopy = (float[,])_x2.Clone();
        float[,] yCopy = (float[,])_y.Clone();
        Random random = new SeededRandom(251207);
        xCopy.PermuteInPlaceTogetherWith(yCopy, random, PermutationUtils.PermuteMethod.Indices, PermutationUtils.CopyMethod.CopyValues);
    }

    // (2) LINQ-based index shuffle (Indices) vs manual Fisher-Yates shuffle
    [Benchmark]
    public void PermuteInPlace_Indices_CopyValues()
    {
        float[,] xCopy = (float[,])_x2.Clone();
        float[,] yCopy = (float[,])_y.Clone();
        Random random = new SeededRandom(251207);
        xCopy.PermuteInPlaceTogetherWith(yCopy, random, PermutationUtils.PermuteMethod.Indices, PermutationUtils.CopyMethod.CopyValues);
    }

    [Benchmark]
    public void PermuteInPlace_FisherYates_CopyValues()
    {
        float[,] xCopy = (float[,])_x2.Clone();
        float[,] yCopy = (float[,])_y.Clone();
        Random random = new SeededRandom(251207);
        xCopy.PermuteInPlaceTogetherWith(yCopy, random, PermutationUtils.PermuteMethod.FisherYates, PermutationUtils.CopyMethod.CopyValues);
    }

    // (3) Per-element 2D indexer loops (CopyValues) vs row-based GetRow/SetRow (SetRow)
    [Benchmark]
    public void PermuteInPlace_Indices_SetRow()
    {
        float[,] xCopy = (float[,])_x2.Clone();
        float[,] yCopy = (float[,])_y.Clone();
        Random random = new SeededRandom(251207);
        xCopy.PermuteInPlaceTogetherWith(yCopy, random, PermutationUtils.PermuteMethod.Indices, PermutationUtils.CopyMethod.SetRow);
    }

    [Benchmark]
    public void PermuteInPlace_FisherYates_SetRow()
    {
        float[,] xCopy = (float[,])_x2.Clone();
        float[,] yCopy = (float[,])_y.Clone();
        Random random = new SeededRandom(251207);
        xCopy.PermuteInPlaceTogetherWith(yCopy, random, PermutationUtils.PermuteMethod.FisherYates, PermutationUtils.CopyMethod.SetRow);
    }*/
}
