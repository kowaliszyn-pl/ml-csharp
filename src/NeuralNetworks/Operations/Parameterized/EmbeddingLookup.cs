// Neural Networks in C♯
// File name: EmbeddingLookup.cs
// www.kowaliszyn.pl, 2025 - 2026

namespace NeuralNetworks.Operations.Parameterized;

/// <summary>
/// Embedding lookup operation that maps integer indices to embedding vectors.
/// </summary>
public class EmbeddingLookup : ParamOperation<int[,], float[,], float[,]>
{
    //private int[,]? _input;

    /// <summary>
    /// Initializes a new instance of the <see cref="EmbeddingLookup"/> class.
    /// </summary>
    /// <param name="embeddings">Embedding matrix [vocabSize x embeddingDim].</param>
    public EmbeddingLookup(float[,] embeddings) : base(embeddings)
    {
    }

    public override string ToString()
        => $"EmbeddingLookup (vocabSize={Param.GetLength(0)}, embeddingDim={Param.GetLength(1)})";

    /// <summary>
    /// Calculates the gradient of the embedding parameters based on the output gradient.
    /// </summary>
    /// <param name="outputGradient">Gradient from upstream [batchSize x (sequenceLength * embeddingDim)].</param>
    /// <returns></returns>
    protected override float[,] CalcParamGradient(float[,] outputGradient)
    {
        int batchSize = outputGradient.GetLength(0);
        int sequenceLength = Input.GetLength(1);

        // Get the dimensions of the embedding matrix
        int embeddingDim = Param.GetLength(1);
        int vocabSize = Param.GetLength(0);

        float[,] paramGradient = new float[vocabSize, embeddingDim];

        // Accumulate gradients
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < sequenceLength; s++)
            {
                int index = Input[b, s];
                for (int e = 0; e < embeddingDim; e++)
                {
                    paramGradient[index, e] += outputGradient[b, s * embeddingDim + e];
                }
            }
        }

        return paramGradient;
    }

    /// <returns>
    /// Embeddings [batchSize x sequenceLength x embeddingDim] flattened to [batchSize x (sequenceLength * embeddingDim)].
    /// </returns>
    protected override float[,] CalcOutput(bool inference)
    {
        int batchSize = Input.GetLength(0);
        int sequenceLength = Input.GetLength(1);
        int embeddingDim = Param.GetLength(1);

        float[,] output = new float[batchSize, sequenceLength * embeddingDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < sequenceLength; s++)
            {
                int index = Input[b, s];
                for (int e = 0; e < embeddingDim; e++)
                {
                    output[b, s * embeddingDim + e] = Param[index, e];
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Calculates the gradient of the input indices based on the output gradient.
    /// </summary>
    /// <param name="outputGradient"></param>
    /// <returns>
    /// Input gradient (not used for integer indices, returns zero array).
    /// </returns>
    protected override int[,] CalcInputGradient(float[,] outputGradient)
        // Return dummy gradient for integer input (not used in backprop)
        => new int[Input.GetLength(0), Input.GetLength(1)];
   
    internal float[,]? GetEmbeddings() 
        => Param;
}