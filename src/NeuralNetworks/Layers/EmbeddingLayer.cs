// Neural Networks in C♯
// File name: EmbeddingLayer.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations.Parameterized;
using NeuralNetworks.ParamInitializers;

namespace NeuralNetworks.Layers;

/// <summary>
/// Embedding layer that maps integer indices to dense vectors.
/// Used in Word2Vec and other NLP models.
/// </summary>
public class EmbeddingLayer : Layer<int[,], float[,]>
{
    private readonly int _vocabSize;
    private readonly int _embeddingDim;
    private readonly ParamInitializer _paramInitializer;

    /// <summary>
    /// Initializes a new instance of the <see cref="EmbeddingLayer"/> class.
    /// </summary>
    /// <param name="vocabSize">Size of the vocabulary (number of unique tokens).</param>
    /// <param name="embeddingDim">Dimensionality of the embedding vectors.</param>
    /// <param name="paramInitializer">Initializer for embedding weights.</param>
    public EmbeddingLayer(int vocabSize, int embeddingDim, ParamInitializer paramInitializer)
    {
        _vocabSize = vocabSize;
        _embeddingDim = embeddingDim;
        _paramInitializer = paramInitializer;
    }

    public override OperationListBuilder<int[,], float[,]> CreateOperationListBuilder()
    {
        // Initialize embedding matrix: [vocabSize x embeddingDim]
        float[,] embeddings = _paramInitializer.InitWeights(_vocabSize, _embeddingDim);

        return AddOperation(new EmbeddingLookup(embeddings));
    }

    public override string ToString()
        => $"EmbeddingLayer (vocabSize={_vocabSize}, embeddingDim={_embeddingDim}, paramInitializer={_paramInitializer})";
}
