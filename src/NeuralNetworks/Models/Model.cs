// Neural Networks in C♯
// File name: Model.cs
// www.kowaliszyn.pl, 2025 - 2026

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
/// <remarks>This class serves as a base for implementing neural network models with customizable layers and loss
/// functions. It supports training workflows, including batch training and parameter optimization, and provides
/// mechanisms for saving and restoring model checkpoints. Derived classes must implement the method for constructing
/// the layer list. Thread safety is not guaranteed; concurrent access should be managed externally.</remarks>
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

    private const int CurrentModelFormatVersion = 1;

    private static readonly JsonSerializerOptions s_paramsSerializerOptions = new()
    {
        WriteIndented = true
    };

    protected Model(LayerListBuilder<TInputData, TPrediction>? layerListBuilder, Loss<TPrediction> lossFunction, SeededRandom? random = null, string? modelFilePath = null)
    {
        LossFunction = lossFunction;
        Random = random;
        _modelFilePath = modelFilePath;
        _layers = (layerListBuilder ?? CreateLayerListBuilderPrivate()).Build();

        if (modelFilePath is not null)
        {
            LoadParams(modelFilePath);
        }
    }

    public Loss<TPrediction> LossFunction { get; private set; }

    protected SeededRandom? Random { get; }

    private protected abstract LayerListBuilder<TInputData, TPrediction> CreateLayerListBuilderPrivate();

    public TPrediction Forward(TInputData input, bool inference)
    {
        RememberInputShape(input);
        return _layers.Forward(input, inference);
    }

    public void Backward(TPrediction lossGrad)
        => _layers.Backward(lossGrad);

    public float TrainBatch(TInputData xBatch, TPrediction yBatch)
    {
        TPrediction predictions = Forward(xBatch, false);
        _lastLoss = LossFunction.Forward(predictions, yBatch);
        Backward(LossFunction.Backward());
        return _lastLoss;
    }

    public void UpdateParams(Optimizer optimizer)
        => _layers.UpdateParams(optimizer);

    public int GetParamCount()
        => _layers.Sum(l => (int?)l.GetParamCount()) ?? 0;

    public List<string> Describe(int indentation = 0)
    {
        string indent = new(' ', indentation);
        string newIndent = new(' ', indentation + Constants.Indentation);
        string layerIndent = new(' ', indentation + 2 * Constants.Indentation);
        List<string> res = [];
        res.Add($"{indent}Model");
        res.Add($"{newIndent}Type: {this}");
        res.Add($"{newIndent}Random: {Random}");
        res.Add($"{newIndent}LossFunction: {LossFunction}");
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
            else if (modelParams.InputShape is not null)
            {
                TInputData syntheticSample = modelParams.InputShape.CreateSyntheticSample<TInputData>(true);
                Forward(syntheticSample, inference: true);
            }
            else
                throw new InvalidOperationException("Model layers are not initialized. Provide an initialization sample via LoadParams(string, TInputData) or ensure the weights file contains input shape metadata.");
        }

        if (_inputShape is null && modelParams.InputShape is not null)
        {
            _inputShape = modelParams.InputShape;
        }

        // Apply parameters to each layer

        for (int layerIndex = 0; layerIndex < _layers.Count; layerIndex++)
        {
            Layer layer = _layers[layerIndex];
            LayerParams layerParams = modelParams.Layers[layerIndex];
            layer.ApplyParams(layerParams, layerIndex);
        }
    }

    private sealed record ModelParams(int Version, List<string> Architecture, string? Comment, ModelInputShape InputShape, List<LayerParams> Layers);

    #endregion Serialization

    #region Checkpoint

    private Model<TInputData, TPrediction>? _checkpoint;

    public void SaveCheckpoint() => _checkpoint = Clone();

    public bool HasCheckpoint() => _checkpoint is not null;

    public void RestoreCheckpoint()
    {
        if (_checkpoint is null)
        {
            throw new Exception("No checkpoint to restore.");
        }
        // _checkpoint is already a deep copy so we can just copy its fields.
        _layers = _checkpoint._layers;
        LossFunction = _checkpoint.LossFunction;
        _lastLoss = _checkpoint._lastLoss;
    }

    /// <summary>
    /// Makes a deep copy of this neural network.
    /// </summary>
    /// <returns></returns>
    public Model<TInputData, TPrediction> Clone()
    {
        Model<TInputData, TPrediction> clone = (Model<TInputData, TPrediction>)MemberwiseClone();
        clone._layers = _layers.Clone();
        clone.LossFunction = LossFunction.Clone();
        return clone;
    }

    #endregion Checkpoint
}
