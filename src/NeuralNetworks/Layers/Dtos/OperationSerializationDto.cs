// Neural Networks in C♯
// File name: OperationWeightsDto.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks.Layers.Dtos;

internal sealed record OperationSerializationDto(string OperationType, ParameterDataDto Parameters);
