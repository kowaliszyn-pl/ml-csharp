// Neural Networks in C♯
// File name: GlobalAveragePooling3DLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System;
using System.Collections.Generic;
using System.Text;

using NeuralNetworks.Layers.OperationList;

namespace NeuralNetworks.Layers;

public class GlobalAveragePooling3DLayer : Layer<float[,,], float[,]>
{
    public override OperationListBuilder<float[,,], float[,]> CreateOperationListBuilder() => throw new NotImplementedException();
}
