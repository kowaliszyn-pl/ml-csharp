// Neural Networks in C♯
// File name: Model.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Text.Json;

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Layers.Dtos;
using NeuralNetworks.Losses;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations;
using NeuralNetworks.Operations.Interfaces;
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
    private ModelInputShapeDto? _inputShape;

    private const int CurrentModelFormatVersion = 1;

    private static readonly JsonSerializerOptions s_paramsSerializerOptions = new()
    {
        WriteIndented = true
    };

    protected Model(LayerListBuilder<TInputData, TPrediction>? layerListBuilder, Loss<TPrediction> lossFunction, SeededRandom? random)
    {
        LossFunction = lossFunction;
        Random = random;
        _layers = (layerListBuilder ?? CreateLayerListBuilderInternal()).Build();
    }

    public IReadOnlyList<Layer> Layers => _layers;

    public Loss<TPrediction> LossFunction { get; private set; }

    protected SeededRandom? Random { get; }

    protected internal abstract LayerListBuilder<TInputData, TPrediction> CreateLayerListBuilderInternal();

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
        res.Add($"{newIndent}Layers");
        foreach (Layer layer in _layers)
        {
            res.Add($"{layerIndent}{layer}");
        }
        return res;
    }

    #region Serialization

    public void SaveParams(string filePath, string? comment = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be empty.", nameof(filePath));

        ModelParams modelParams = GetModelParams(comment);

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
        ModelParams? dto = JsonSerializer.Deserialize<ModelParams>(json, s_paramsSerializerOptions) 
            ?? throw new InvalidOperationException("Unable to deserialize params file.");

        EnsureLayersInitialized(initializationSample, dto.InputShape);

        if (_inputShape is null && dto.InputShape is not null)
        {
            _inputShape = dto.InputShape;
        }

        ApplyWeights(dto);
    }

    private void EnsureLayersInitialized(TInputData? initializationSample, ModelInputShapeDto? persistedInputShape)
    {
        if (_layers.All(layer => layer.IsInitialized))
            return;

        if (initializationSample is not null)
        {
            Forward(initializationSample, inference: true);
            return;
        }

        if (persistedInputShape is not null)
        {
            TInputData syntheticSample = CreateInputFromShape(persistedInputShape);
            Forward(syntheticSample, inference: true);
            return;
        }

        throw new InvalidOperationException("Model layers are not initialized. Provide an initialization sample via LoadParams(string, TInputData) or ensure the weights file contains input shape metadata.");
    }

    private ModelParams GetModelParams(string? comment)
    {
        List<LayerSerializationDto> serializedLayers = _layers
            .Select(layer => layer.Serialize())
            .ToList();

        ModelInputShapeDto inputShape = _inputShape
            ?? throw new InvalidOperationException("Unable to determine model input shape. Run a forward pass before saving parameters.");

        return new ModelParams(CurrentModelFormatVersion, Describe(), comment, serializedLayers, inputShape);
    }

    private void ApplyWeights(ModelParams dto)
    {
        if (dto.Version != CurrentModelFormatVersion)
            throw new InvalidOperationException($"Unsupported weights file version '{dto.Version}'. Expected '{CurrentModelFormatVersion}'.");

        if (dto.Layers.Count != _layers.Count)
            throw new InvalidOperationException($"Layer count mismatch. Model has {_layers.Count} layers but weights file has {dto.Layers.Count}.");

        for (int layerIndex = 0; layerIndex < _layers.Count; layerIndex++)
        {
            Layer layer = _layers[layerIndex];
            LayerSerializationDto layerDto = dto.Layers[layerIndex];

            EnsureTypeMatch(layerDto.LayerType, layer.GetType(), layerIndex, operationIndex: null);

            IReadOnlyList<Operation> operations = layer.GetOperations();
            List<Operation> parameterizedOperations = operations.Where(op => op is IParamOperation).ToList();

            if (parameterizedOperations.Count != layerDto.Operations.Count)
            {
                throw new InvalidOperationException($"Operation count mismatch for layer '{layer.GetType().Name}' at index {layerIndex}. Expected {parameterizedOperations.Count} operations with parameters but file has {layerDto.Operations.Count}.");
            }

            for (int opIndex = 0; opIndex < parameterizedOperations.Count; opIndex++)
            {
                Operation operation = parameterizedOperations[opIndex];
                OperationSerializationDto operationDto = layerDto.Operations[opIndex];

                EnsureTypeMatch(operationDto.OperationType, operation.GetType(), layerIndex, opIndex);

                IParamOperation provider = (IParamOperation)operation;
                provider.Restore(operationDto.Parameters.ToSnapshot());
            }
        }
    }

    private static void EnsureTypeMatch(string? persistedType, Type runtimeType, int layerIndex, int? operationIndex)
    {
        string expectedType = GetTypeIdentifier(runtimeType);
        if (!string.Equals(persistedType, expectedType, StringComparison.Ordinal))
        {
            string location = operationIndex is null
                ? $"Layer {layerIndex}"
                : $"Layer {layerIndex}, operation {operationIndex}";
            throw new InvalidOperationException($"Type mismatch at {location}. Expected '{expectedType}' but found '{persistedType ?? "<unknown>"}'.");
        }
    }

    private void RememberInputShape(TInputData input)
    {
        ModelInputShapeDto? capturedShape = TryCaptureInputShape(input);
        if (capturedShape is not null)
        {
            _inputShape = capturedShape;
        }
    }

    private static ModelInputShapeDto? TryCaptureInputShape(TInputData input)
    {
        if (input is not Array array)
            return null;

        int[] dimensions = new int[array.Rank];
        for (int dimension = 0; dimension < array.Rank; dimension++)
        {
            dimensions[dimension] = array.GetLength(dimension);
        }

        return new ModelInputShapeDto(GetTypeIdentifier(typeof(TInputData)), dimensions);
    }

    private static TInputData CreateInputFromShape(ModelInputShapeDto inputShape)
    {
        if (inputShape is null)
            throw new ArgumentNullException(nameof(inputShape));

        string expectedType = GetTypeIdentifier(typeof(TInputData));
        if (!string.Equals(inputShape.InputType, expectedType, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Input type mismatch. Model expects '{expectedType}' but file contains '{inputShape.InputType ?? "<unknown>"}'.");
        }

        if (!typeof(TInputData).IsArray)
        {
            throw new NotSupportedException($"Input type '{typeof(TInputData)}' is not supported for shape-based initialization. Provide an initialization sample explicitly.");
        }

        Type? elementType = typeof(TInputData).GetElementType();
        if (elementType is null)
            throw new InvalidOperationException($"Unable to determine element type for '{typeof(TInputData)}'.");

        if (inputShape.Dimensions is null || inputShape.Dimensions.Length == 0)
            throw new InvalidOperationException("Persisted input shape is empty.");

        Array sample = Array.CreateInstance(elementType, inputShape.Dimensions);
        return (TInputData)(object)sample;
    }

    private sealed record ModelParams(int Version, List<string> Architecture, string? Comment, List<LayerSerializationDto> Layers, ModelInputShapeDto? InputShape);

    private sealed record ModelInputShapeDto(string InputType, int[] Dimensions);

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
