// Neural Networks in C♯
// File name: Model.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;
using System.Text;
using System.Text.Json;

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Utils.ModelUtils;

namespace NeuralNetworks.Models;

/// <summary>
/// Represents an abstract neural network model that processes input data and produces predictions. Provides core
/// functionality for forward and backward passes, training, parameter updates, and checkpointing.
/// </summary>
/// <remarks>
/// This class serves as a base for implementing neural network models with customizable layers and loss functions. It
/// supports training workflows, including batch training and parameter optimization, and provides mechanisms for saving
/// and restoring model checkpoints. Derived classes must implement the method for constructing the layer list. Thread
/// safety is not guaranteed; concurrent access should be managed externally.
/// </remarks>
/// <typeparam name="TInputData">The type of input data provided to the model. Must not be null.</typeparam>
/// <typeparam name="TPrediction">The type of prediction output produced by the model. Must not be null.</typeparam>
public abstract class Model<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    private LayerList<TInputData, TPrediction> _layers;
    private float _lastLoss; 
    private readonly string? _modelFilePath;
    private ModelInputShape? _inputShape;
    private readonly Loss<TPrediction>? _defaultLossFunction;

    private const int CurrentModelFormatVersion = 1;

    private static readonly JsonSerializerOptions s_paramsSerializerOptions = new()
    {
        WriteIndented = true
    };

    protected Model(LayerListBuilder<TInputData, TPrediction>? layerListBuilder, Loss<TPrediction>? defaultLossFunction = null, SeededRandom? random = null, string? modelFilePath = null)
    {
        _defaultLossFunction = defaultLossFunction;
        Random = random;
        _modelFilePath = modelFilePath;
        _layers = (layerListBuilder ?? CreateLayerListBuilderPrivate()).Build();

        if (modelFilePath is not null)
        {
            LoadParams(modelFilePath);
        }
    }

    protected SeededRandom? Random { get; }

    //protected List<Layer> Layers => _layers;

    private protected abstract LayerListBuilder<TInputData, TPrediction> CreateLayerListBuilderPrivate();

    public TPrediction Forward(TInputData input, bool inference)
    {
        RememberInputShape(input);
        return _layers.Forward(input, inference);
    }

    protected TPrediction InferenceFromLayer<TLayerInputData>(Layer fromLayer, TLayerInputData input)
        where TLayerInputData : notnull
    {
        object stream = input;
        int indexOfFirstDecoderLayer = _layers.IndexOf(fromLayer);

        Debug.Assert(indexOfFirstDecoderLayer >= 0, "The specified layer was not found in the model's layer list.");

        foreach (Layer layer in _layers.Skip(indexOfFirstDecoderLayer))
        {
            stream = layer.Forward(stream, inference: true);
        }
        return (TPrediction)stream;
    }

    public void Backward(TPrediction lossGrad)
        => _layers.Backward(lossGrad);

    public float TrainBatch(TInputData xBatch, TPrediction yBatch, Loss<TPrediction>? lossFunction = null)
    {
        lossFunction = ResolveLossFunction(lossFunction);

        TPrediction predictions = Forward(xBatch, false);
        _lastLoss = lossFunction.Forward(predictions, yBatch);
        Backward(lossFunction.Backward());
        return _lastLoss;
    }

    public float CalculateLoss(TPrediction predictions, TPrediction targets, Loss<TPrediction>? lossFunction = null)
    {
        lossFunction = ResolveLossFunction(lossFunction);
        return lossFunction.Forward(predictions, targets);
    }

    private Loss<TPrediction> ResolveLossFunction(Loss<TPrediction>? lossFunction)
    {
        lossFunction ??= _defaultLossFunction;

        Debug.Assert(lossFunction != null, "A loss function must be provided either via the method parameter or as the model's default loss function.");

        return lossFunction;
    }

    public void UpdateParams(Optimizer optimizer)
        => _layers.UpdateParams(optimizer);

    public int GetParamCount()
        => _layers.Sum(l => (int?)l.GetParamCount()) ?? 0;

    public virtual List<string> Describe(int indentation = 0)
    {
        string indent = new(' ', indentation);
        string newIndent = new(' ', indentation + Constants.Indentation);
        string layerIndent = new(' ', indentation + 2 * Constants.Indentation);
        List<string> res = [];
        res.Add($"{indent}Model");
        res.Add($"{newIndent}Type: {this}");
        res.Add($"{newIndent}Random: {Random}");
        res.Add($"{newIndent}DefaultLossFunction: {_defaultLossFunction}");
        if (_modelFilePath != null)
            res.Add($"{newIndent}ModelFilePath: {_modelFilePath}");
        res.Add($"{newIndent}Layers");
        foreach (Layer layer in _layers)
        {
            res.Add($"{layerIndent}{layer}");
        }
        return res;
    }

    private void RememberInputShape(TInputData input)
    {
        // If input shape is already known, no need to capture it again.
        if (_inputShape != null || input is not Array array)
            return;

        int[] dimensions = new int[array.Rank];
        for (int dimension = 0; dimension < array.Rank; dimension++)
        {
            dimensions[dimension] = array.GetLength(dimension);
        }

        _inputShape = new ModelInputShape(GetTypeIdentifier(typeof(TInputData)), dimensions);
    }

    #region Serialization

    public void SaveParams(string filePath, string? comment = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be empty.", nameof(filePath));

        // Get the model parameters to serialize

        List<LayerParams> serializedLayers = _layers
            .Select(layer => layer.GetParams())
            .ToList();

        ModelInputShape inputShape = _inputShape
            ?? throw new InvalidOperationException("Unable to determine model input shape. Run a forward pass before saving parameters.");

        ModelParams modelParams = new(CurrentModelFormatVersion, Describe(), comment, inputShape, serializedLayers);

        // Save to file

        string? directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string json = JsonSerializer.Serialize(modelParams, s_paramsSerializerOptions);
        File.WriteAllText(filePath, json);
    }

    public void LoadParams(string filePath, TInputData? initializationSample = default)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be empty.", nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException("Params file was not found.", filePath);

        string json = File.ReadAllText(filePath);
        ModelParams modelParams = JsonSerializer.Deserialize<ModelParams>(json, s_paramsSerializerOptions)
            ?? throw new InvalidOperationException("Unable to deserialize params file.");

        if (modelParams.Version != CurrentModelFormatVersion)
            throw new InvalidOperationException($"Unsupported weights file version '{modelParams.Version}'. Expected '{CurrentModelFormatVersion}'.");

        if (modelParams.Layers.Count != _layers.Count)
            throw new InvalidOperationException($"Layer count mismatch. Model has {_layers.Count} layers but weights file has {modelParams.Layers.Count}.");

        // Ensure all layers are initialized before applying weights

        if (_layers.Any(layer => !layer.IsInitialized))
        {
            if (initializationSample is not null)
            {
                Forward(initializationSample, inference: true);
            }
            else if (modelParams.Input is not null)
            {
                TInputData syntheticSample = modelParams.Input.CreateSyntheticSample<TInputData>(true);
                Forward(syntheticSample, inference: true);
            }
            else
                throw new InvalidOperationException("Model layers are not initialized. Provide an initialization sample via LoadParams(string, TInputData) or ensure the weights file contains input shape metadata.");
        }

        if (_inputShape is null && modelParams.Input is not null)
        {
            _inputShape = modelParams.Input;
        }

        // Apply parameters to each layer

        for (int layerIndex = 0; layerIndex < _layers.Count; layerIndex++)
        {
            Layer layer = _layers[layerIndex];
            LayerParams layerParams = modelParams.Layers[layerIndex];
            layer.ApplyParams(layerParams, layerIndex);
        }
    }

    private sealed record ModelParams(int Version, List<string> ArchitectureDescription, string? Comment, ModelInputShape Input, List<LayerParams> Layers);

    #endregion Serialization
}
