// Neural Networks in C?
// File name: Gpt2ParameterLoader.cs
// www.kowaliszyn.pl, 2026

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NeuralNetworks.Transformers.Gpt2;

public static class Gpt2ParameterLoader
{
    private const string Magic = "GPT2WEIGHTS";
    private const int FormatVersion = 1;
    private static readonly byte[] MagicBytes = Encoding.ASCII.GetBytes(Magic);

    public static Gpt2Parameters LoadFromFile(string filePath, Gpt2Config config)
    {
        ArgumentException.ThrowIfNullOrEmpty(filePath);
        ArgumentNullException.ThrowIfNull(config);

        using FileStream stream = File.OpenRead(filePath);
        return LoadFromStream(stream, config);
    }

    public static Gpt2Parameters LoadFromStream(Stream stream, Gpt2Config config)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(config);

        Dictionary<string, Tensor> tensors = ReadTensorTable(stream);
        float[,] tokenEmbeddings = ToMatrix(GetTensor(tensors, "token_embeddings"));
        float[,] positionalEmbeddings = ToMatrix(GetTensor(tensors, "positional_embeddings"));

        TransformerBlockParameters[] blocks = new TransformerBlockParameters[config.LayerCount];
        for (int i = 0; i < blocks.Length; i++)
            blocks[i] = BuildBlock(tensors, i);

        LayerNormParameters finalLayerNorm = new(
            ToVector(GetTensor(tensors, "final_layer_norm.gamma")),
            ToVector(GetTensor(tensors, "final_layer_norm.beta")));

        return new Gpt2Parameters(tokenEmbeddings, positionalEmbeddings, blocks, finalLayerNorm);
    }

    private static TransformerBlockParameters BuildBlock(Dictionary<string, Tensor> tensors, int blockIndex)
    {
        string prefix = $"blocks.{blockIndex}.";
        LinearWeights projection = new(
            ToMatrix(GetTensor(tensors, $"{prefix}attn.qkv.weight")),
            ToVector(GetTensor(tensors, $"{prefix}attn.qkv.bias")));
        LinearWeights outputProjection = new(
            ToMatrix(GetTensor(tensors, $"{prefix}attn.out.weight")),
            ToVector(GetTensor(tensors, $"{prefix}attn.out.bias")));
        LinearWeights up = new(
            ToMatrix(GetTensor(tensors, $"{prefix}mlp.up.weight")),
            ToVector(GetTensor(tensors, $"{prefix}mlp.up.bias")));
        LinearWeights down = new(
            ToMatrix(GetTensor(tensors, $"{prefix}mlp.down.weight")),
            ToVector(GetTensor(tensors, $"{prefix}mlp.down.bias")));

        LayerNormParameters ln1 = new(
            ToVector(GetTensor(tensors, $"{prefix}ln1.gamma")),
            ToVector(GetTensor(tensors, $"{prefix}ln1.beta")));
        LayerNormParameters ln2 = new(
            ToVector(GetTensor(tensors, $"{prefix}ln2.gamma")),
            ToVector(GetTensor(tensors, $"{prefix}ln2.beta")));

        return new TransformerBlockParameters(
            new MultiHeadAttentionParameters(projection, outputProjection),
            new FeedForwardParameters(up, down),
            ln1,
            ln2);
    }

    private static Tensor GetTensor(Dictionary<string, Tensor> tensors, string name)
    {
        if (!tensors.TryGetValue(name, out Tensor? tensor))
            throw new InvalidDataException($"Tensor '{name}' was not found in the weight file.");
        return tensor;
    }

    private static Dictionary<string, Tensor> ReadTensorTable(Stream stream)
    {
        using BinaryReader reader = new(stream, Encoding.UTF8, leaveOpen: true);
        byte[] magic = reader.ReadBytes(MagicBytes.Length);
        if (magic.Length != MagicBytes.Length || !magic.AsSpan().SequenceEqual(MagicBytes))
            throw new InvalidDataException("Unsupported GPT-2 weight file (invalid magic header).");

        int version = reader.ReadInt32();
        if (version != FormatVersion)
            throw new InvalidDataException($"Unsupported GPT-2 weight file version {version}.");

        int tensorCount = reader.ReadInt32();
        if (tensorCount <= 0)
            throw new InvalidDataException("Tensor count must be positive.");

        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        for (int i = 0; i < tensorCount; i++)
        {
            string name = ReadName(reader);
            Tensor tensor = ReadTensor(reader, name);
            tensors[name] = tensor;
        }

        return tensors;
    }

    private static string ReadName(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        if (length <= 0)
            throw new InvalidDataException("Tensor name length must be positive.");

        byte[] bytes = reader.ReadBytes(length);
        if (bytes.Length != length)
            throw new EndOfStreamException("Unexpected end of stream while reading tensor name.");

        return Encoding.UTF8.GetString(bytes);
    }

    private static Tensor ReadTensor(BinaryReader reader, string name)
    {
        int rank = reader.ReadInt32();
        if (rank <= 0)
            throw new InvalidDataException($"Tensor '{name}' rank must be positive.");

        int[] shape = new int[rank];
        long elementCount = 1;
        for (int i = 0; i < rank; i++)
        {
            int dimension = reader.ReadInt32();
            if (dimension <= 0)
                throw new InvalidDataException($"Tensor '{name}' dimensions must be positive.");
            shape[i] = dimension;
            elementCount = checked(elementCount * dimension);
        }

        if (elementCount <= 0 || elementCount > int.MaxValue)
            throw new InvalidDataException($"Tensor '{name}' is too large.");

        float[] data = new float[elementCount];
        for (int i = 0; i < data.Length; i++)
            data[i] = reader.ReadSingle();

        return new Tensor(name, shape, data);
    }

    private static float[,] ToMatrix(Tensor tensor)
    {
        if (tensor.Shape.Length != 2)
            throw new InvalidDataException($"Tensor '{tensor.Name}' must be 2D.");

        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];
        float[,] matrix = new float[rows, cols];
        Buffer.BlockCopy(tensor.Data, 0, matrix, 0, tensor.Data.Length * sizeof(float));
        return matrix;
    }

    private static float[] ToVector(Tensor tensor)
    {
        if (tensor.Shape.Length != 1)
            throw new InvalidDataException($"Tensor '{tensor.Name}' must be 1D.");
        return tensor.Data;
    }

    private sealed record Tensor(string Name, int[] Shape, float[] Data);
}
