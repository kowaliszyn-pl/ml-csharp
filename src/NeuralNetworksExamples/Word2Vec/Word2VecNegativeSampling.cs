// Neural Networks in C♯
// File name: Word2VecNegativeSampling.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Models;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations.ActivationFunctions;
using NeuralNetworks.ParamInitializers;

namespace NeuralNetworksExamples.Word2Vec;

// TODO: Add a Word2VecNegativeSamplingModel class (in new file) that uses negative sampling loss instead of softmax loss for more efficient training on large vocabularies.
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
}

internal static class Word2VecNegativeSampling
{
    public static void Run()
    {
    }
}
