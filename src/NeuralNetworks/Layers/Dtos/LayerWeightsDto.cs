// Neural Networks in C♯
// File name: LayerWeightsDto.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks.Layers.Dtos;

internal sealed record LayerWeightsDto(string LayerType, List<OperationWeightsDto> Operations);
