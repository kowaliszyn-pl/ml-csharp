// Neural Networks in C♯
// File name: EmbeddingLookup.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

namespace NeuralNetworks.Operations.Parameterized;

/// <summary>
/// Embedding lookup operation that maps integer indices to embedding vectors.
/// </summary>
/// <remarks>
/// Initializes a new instance of the <see cref="EmbeddingLookup"/> class.
/// </remarks>
/// <param name="embeddings">Embedding matrix [vocabSize, embeddingDim].</param>
public class EmbeddingLookup(float[,] embeddings) : ParamOperation<int[,], float[,], float[,]>(embeddings)
{
    /// <summary>
    /// Calculates the gradient of the embedding parameters based on the output gradient.
    /// </summary>
    /// <param name="outputGradient">Gradient from upstream [batchSize, sequenceLength * embeddingDim].</param>
    /// <returns></returns>
    protected override float[,] CalcParamGradient(float[,] outputGradient)
    {
        int batchSize = Input.GetLength(0);
        int sequenceLength = Input.GetLength(1);

        // Get the dimensions of the embedding matrix
        int embeddingDim = Param.GetLength(1);
        int vocabSize = Param.GetLength(0);

        Debug.Assert(batchSize == outputGradient.GetLength(0), "Batch size mismatch between input and output gradient.");

        float[,] paramGradient = new float[vocabSize, embeddingDim];

        // Accumulate gradients
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < sequenceLength; s++)
            {
                int tokenId = Input[b, s];

                Debug.Assert(tokenId >= 0 && tokenId < vocabSize, $"Index out of range: {tokenId}");

                for (int e = 0; e < embeddingDim; e++)
                {
                    paramGradient[tokenId, e] += outputGradient[b, s * embeddingDim + e];
                }
            }
        }

        return paramGradient;
    }

    /// <returns>
    /// Embeddings [batchSize, sequenceLength, embeddingDim] flattened to [batchSize, sequenceLength * embeddingDim].
    /// </returns>
    protected override float[,] CalcOutput(bool inference)
    {
        int batchSize = Input.GetLength(0);
        int sequenceLength = Input.GetLength(1);

        int vocabSize = Param.GetLength(0);
        int embeddingDim = Param.GetLength(1);

        float[,] output = new float[batchSize, sequenceLength * embeddingDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < sequenceLength; s++)
            {
                int tokenId = Input[b, s];

                Debug.Assert(tokenId >= 0 && tokenId < vocabSize, $"Index out of range: {tokenId}");

                for (int e = 0; e < embeddingDim; e++)
                {
                    output[b, s * embeddingDim + e] = Param[tokenId, e];
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

    public override string ToString()
        => $"EmbeddingLookup (vocabSize={Param.GetLength(0)}, embeddingDim={Param.GetLength(1)})";
}