// Neural Networks in C♯
// File name: SpanBenchmarks.cs
// www.kowaliszyn.pl, 2025

using System;

using BenchmarkDotNet.Attributes;

using Microsoft.VSDiagnostics;

namespace SpanBenchmark;
// For more information on the VS BenchmarkDotNet Diagnosers see https://learn.microsoft.com/visualstudio/profiling/profiling-with-benchmark-dotnet
[CPUUsageDiagnoser]
public class SpanBenchmarks
{
    private float[,,,] array1;
    private float[,,,] array2;

    [GlobalSetup]
    public void Setup()
    {
        array1 = NeuralNetworks.Core.ArrayUtils.CreateRandom(100, 100, 3, 3, new Random(42));
        array2 = NeuralNetworks.Core.ArrayUtils.CreateRandom(100, 100, 3, 3, new Random(43));
    }

    [Benchmark]
    public void MultiplyByTanhDerivativeSpan()
    {
        //new NeuralNetworks.Core.Span.OperationOps.MultiplyByTanhDerivative(array1, array2);
    }

    [Benchmark]
    public void MultiplyByTanhDerivative()
    {
        NeuralNetworks.Core.ArrayExtensions.MultiplyByTanhDerivative(array1, array2);
    }

    [Benchmark]
    public void TanhSpan()
    {
        //NeuralNetworks.Core.Span.ArrayExtensions.Tanh(array1);
    }

    [Benchmark]
    public void Tanh()
    {
        //NeuralNetworks.Core.ArrayExtensions.Tanh(array1);
    }
}
