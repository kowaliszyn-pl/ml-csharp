// Neural Networks in C♯
// File name: LayerParams.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Operations.Parameterized;

namespace NeuralNetworks.Layers;

internal sealed record LayerParams(string LayerType, List<ParamOperationData> Operations);
