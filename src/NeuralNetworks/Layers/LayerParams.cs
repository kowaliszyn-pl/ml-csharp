// Neural Networks in C♯
// File name: LayerParams.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers.Dtos;

namespace NeuralNetworks.Layers;

internal sealed record LayerParams(string LayerType, List<OperationSerializationDto> Operations);
