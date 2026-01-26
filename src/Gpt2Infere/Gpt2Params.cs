// Neural Networks in C♯
// File name: Gpt2Params.cs
// www.kowaliszyn.pl, 2025 - 2026

// Translated for C# from the original Python code at https://github.com/jaymody/picoGPT (fork https://github.com/kowaliszyn-pl/pico-gpt-2)
// Also, part of the code also copied from https://github.com/lofcz/gpt2sharp (fork https://github.com/kowaliszyn-pl/sharp-gpt-2)

using System.Numerics.Tensors;
using System.Text;

using static NeuralNetworks.Core.RandomUtils;

internal sealed record Gpt2Params
{
    public float[,] TokenEmbeddings { get; init; } = default!;
    public float[,] PositionalEmbeddings { get; init; } = default!;
    public Gpt2Block[] Blocks { get; init; } = default!;
    public Gpt2LayerNormParams FinalLayerNorm { get; init; } = default!;

    public static Gpt2Params CreateNew(Gpt2HParams gpt2HParams, Random random)
    {
        return new()
        {
            TokenEmbeddings = CreateRandomNormal(gpt2HParams.VocabularySize, gpt2HParams.EmbeddingSize, random),
            PositionalEmbeddings = CreateRandomNormal(gpt2HParams.ContextSize, gpt2HParams.EmbeddingSize, random),
            Blocks = Enumerable
                    .Range(0, gpt2HParams.LayerCount)
                    .Select(_ => new Gpt2Block()
                    {
                        LayerNorm1 = new Gpt2LayerNormParams
                        {
                            Gamma = CreateRandomNormal(gpt2HParams.EmbeddingSize, random),
                            Beta = CreateRandomNormal(gpt2HParams.EmbeddingSize, random)
                        },
                        LayerNorm2 = new Gpt2LayerNormParams
                        {
                            Gamma = CreateRandomNormal(gpt2HParams.EmbeddingSize, random),
                            Beta = CreateRandomNormal(gpt2HParams.EmbeddingSize, random)
                        },
                        Attention = new Gpt2MultiHeadAttentionParams
                        {
                            Projection = new Gpt2LinearParams
                            {
                                Weights = CreateRandomNormal(gpt2HParams.EmbeddingSize, 3 * gpt2HParams.EmbeddingSize, random),
                                Bias = CreateRandomNormal(3 * gpt2HParams.EmbeddingSize, random)
                            },
                            OutputProjection = new Gpt2LinearParams
                            {
                                Weights = CreateRandomNormal(gpt2HParams.EmbeddingSize, gpt2HParams.EmbeddingSize, random),
                                Bias = CreateRandomNormal(gpt2HParams.EmbeddingSize, random)
                            }
                        },
                        MultiLayerPerceptron = new Gpt2MultiLayerPerceptron
                        {
                            FullyConnected = new Gpt2LinearParams
                            {
                                Weights = CreateRandomNormal(gpt2HParams.EmbeddingSize, 4 * gpt2HParams.EmbeddingSize, random),
                                Bias = CreateRandomNormal(4 * gpt2HParams.EmbeddingSize, random)
                            },
                            OutputProjection = new Gpt2LinearParams
                            {
                                Weights = CreateRandomNormal(4 * gpt2HParams.EmbeddingSize, gpt2HParams.EmbeddingSize, random),
                                Bias = CreateRandomNormal(gpt2HParams.EmbeddingSize, random)
                            }
                        }
                    }
                    )
                    .ToArray(),
            FinalLayerNorm = new Gpt2LayerNormParams
            {
                Gamma = CreateRandomNormal(gpt2HParams.EmbeddingSize, random),
                Beta = CreateRandomNormal(gpt2HParams.EmbeddingSize, random)
            }
        };
    }

    private sealed record Tensor(string Name, int[] Shape, float[] As1D)
    {
        public float[,] As2D
        {
            get
            {
                int rows = Shape[0];
                int cols = Shape[1];
                float[,] matrix = new float[rows, cols];
                Buffer.BlockCopy(As1D, 0, matrix, 0, As1D.Length * sizeof(float));
                return matrix;
            }
        }
    }

    internal static Gpt2Params FromDirectory(string modelDirectory, Gpt2HParams gpt2HParams)
    {
        const string Magic = "GPT2WEIGHTS";
        const int FormatVersion = 1;
        byte[] magicBytes = Encoding.ASCII.GetBytes(Magic);

        string path = Path.Combine(modelDirectory, "weights.bin");
        using FileStream stream = File.OpenRead(path);

        // Read tensor table
        using BinaryReader reader = new(stream, Encoding.UTF8, leaveOpen: true);
        byte[] magic = reader.ReadBytes(magicBytes.Length);
        if (magic.Length != magicBytes.Length || !magic.AsSpan().SequenceEqual(magicBytes))
            throw new InvalidDataException("Unsupported GPT-2 weight file (invalid magic header).");

        int version = reader.ReadInt32();
        if (version != FormatVersion)
            throw new InvalidDataException($"Unsupported GPT-2 weight file version {version}.");

        int tensorCount = reader.ReadInt32();

        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        for (int i = 0; i < tensorCount; i++)
        {
            string name = ReadName(reader);
            Tensor tensor = ReadTensor(reader, name);
            tensors[name] = tensor;
        }

        Gpt2Params gpt2Params = new()
        {
            TokenEmbeddings = tensors["token_embeddings"].As2D,
            PositionalEmbeddings = tensors["positional_embeddings"].As2D,
            FinalLayerNorm = new Gpt2LayerNormParams
            {
                Gamma = tensors["final_layer_norm.gamma"].As1D,
                Beta = tensors["final_layer_norm.beta"].As1D
            },
            Blocks = Enumerable.Range(0, gpt2HParams.LayerCount).Select(i =>
            {
                string prefix = $"blocks.{i}.";
                return new Gpt2Block
                {
                    LayerNorm1 = new Gpt2LayerNormParams
                    {
                        Gamma = tensors[$"{prefix}ln1.gamma"].As1D,
                        Beta = tensors[$"{prefix}ln1.beta"].As1D
                    },
                    Attention = new Gpt2MultiHeadAttentionParams
                    {
                        Projection = new Gpt2LinearParams
                        {
                            Weights = tensors[$"{prefix}attn.qkv.weight"].As2D,
                            Bias = tensors[$"{prefix}attn.qkv.bias"].As1D
                        },
                        OutputProjection = new Gpt2LinearParams
                        {
                            Weights = tensors[$"{prefix}attn.out.weight"].As2D,
                            Bias = tensors[$"{prefix}attn.out.bias"].As1D
                        }
                    },
                    LayerNorm2 = new Gpt2LayerNormParams
                    {
                        Gamma = tensors[$"{prefix}ln2.gamma"].As1D,
                        Beta = tensors[$"{prefix}ln2.beta"].As1D
                    },
                    MultiLayerPerceptron = new Gpt2MultiLayerPerceptron
                    {
                        FullyConnected = new Gpt2LinearParams
                        {
                            Weights = tensors[$"{prefix}mlp.up.weight"].As2D,
                            Bias = tensors[$"{prefix}mlp.up.bias"].As1D
                        },
                        OutputProjection = new Gpt2LinearParams
                        {
                            Weights = tensors[$"{prefix}mlp.down.weight"].As2D,
                            Bias = tensors[ $"{prefix}mlp.down.bias"].As1D
                        }
                    }
                };
            }).ToArray()
        };

        return gpt2Params;

        static string ReadName(BinaryReader reader)
        {
            int length = reader.ReadInt32();
            byte[] bytes = reader.ReadBytes(length);
            return Encoding.UTF8.GetString(bytes);
        }

        static Tensor ReadTensor(BinaryReader reader, string name)
        {
            // First, read the rank
            int rank = reader.ReadInt32();

            // Then, read the shape
            int[] shape = new int[rank];
            long elementCount = 1;
            for (int i = 0; i < rank; i++)
            {
                int dimension = reader.ReadInt32();
                shape[i] = dimension;
                elementCount = checked(elementCount * dimension);
            }

            // Finally, read the data
            float[] data = new float[elementCount];
            for (int i = 0; i < data.Length; i++)
                data[i] = reader.ReadSingle();

            return new Tensor(name, shape, data);
        }
    }
}

internal sealed record Gpt2Block
{
    public Gpt2LayerNormParams LayerNorm1 { get; init; } = default!;
    public Gpt2MultiHeadAttentionParams Attention { get; init; } = default!;
    public Gpt2LayerNormParams LayerNorm2 { get; init; } = default!;
    public Gpt2MultiLayerPerceptron MultiLayerPerceptron { get; init; } = default!;
}

internal sealed record Gpt2LayerNormParams
{
    public float[] Gamma { get; init; } = default!;
    public float[] Beta { get; init; } = default!;
}

internal sealed record Gpt2MultiHeadAttentionParams
{
    public Gpt2LinearParams Projection { get; init; } = default!;
    public Gpt2LinearParams OutputProjection { get; init; } = default!;
}

internal sealed record Gpt2LinearParams
{
    public float[,] Weights { get; init; } = default!;
    public float[] Bias { get; init; } = default!;
}

internal sealed record Gpt2MultiLayerPerceptron
{
    /// <summary>
    /// Multi layer perceptron, full connected
    /// </summary>
    public Gpt2LinearParams FullyConnected { get; init; } = default!;

    /// <summary>
    /// Multi layer perceptron, projection
    /// </summary>
    public Gpt2LinearParams OutputProjection { get; init; } = default!;
}