// Neural Networks in C♯
// File name: ModelParams.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Text.Json.Serialization;

using NeuralNetworks.Layers;

namespace NeuralNetworks.Models;

internal sealed record ModelParams
{
    public int Version { get; }
    public DateTime Created { get; }
    public List<string> ArchitectureDescription { get; }
    public string? Comment { get; }
    public ModelInputShape Input { get; }
    public List<LayerParams> Layers { get; }

    public ModelParams(int version, List<string> architectureDescription, string? comment, ModelInputShape input, List<LayerParams> layers)
        : this(version, architectureDescription, comment, input, layers, DateTime.Now)
    {
    }

    [JsonConstructor]
    public ModelParams(int version, List<string> architectureDescription, string? comment, ModelInputShape input, List<LayerParams> layers, DateTime created)
    {
        Version = version;
        Created = created;
        ArchitectureDescription = architectureDescription;
        Comment = comment;
        Input = input;
        Layers = layers;
    }
}
