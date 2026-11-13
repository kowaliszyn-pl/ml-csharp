// Neural Networks in C♯
// File name: EmbeddingDotProductLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.ParamInitializers;
using NeuralNetworks.Regularization;

namespace NeuralNetworks.Layers;

internal class EmbeddingDotProductLayer(int vocabSize, int embeddingDim, ParamInitializer paramInitializer, IRegularizer? regularizer = null)
    : Layer<int[,], float[,]>
{
    public override OperationListBuilder<int[,], float[,]> CreateOperationListBuilder() => throw new NotImplementedException();
}
