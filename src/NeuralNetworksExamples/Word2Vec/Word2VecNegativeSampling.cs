// Neural Networks in C♯
// File name: Word2VecNegativeSampling.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.ParamInitializers;

namespace NeuralNetworksExamples.Word2Vec;

internal class Word2VecNegativeSamplingModel(int vocabSize, int embeddingDim, SeededRandom? random = null)
    : BaseModel<int[,], float[,]>(new SigmoidBinaryCrossEntropyLoss(), random)
{
    private EmbeddingLayer? _embeddingLayer;

    protected override LayerListBuilder<int[,], float[,]> CreateLayerListBuilder()
    {
        GlorotInitializer initializer = new(Random);
        return
            AddLayer(_embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim, initializer))
            .AddLayer(new DenseLayer(1, new Linear(), initializer));
    }

    public float[,] GetEmbeddings()
    {
        Debug.Assert(_embeddingLayer != null, "_embeddingLayer should not be null here.");
        return _embeddingLayer.GetEmbeddings();
    }
}

internal static class Word2VecNegativeSampling
{
    public static void Run()
    {
    }
}
