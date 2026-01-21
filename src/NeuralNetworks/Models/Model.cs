// Neural Networks in C♯
// File name: Model.cs
// www.kowaliszyn.pl, 2025

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Models.LayerList;
using NeuralNetworks.Operations;
using NeuralNetworks.Operations.Interfaces;
using NeuralNetworks.Optimizers;

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
    private const int CurrentWeightsFormatVersion = 1;
    private static readonly JsonSerializerOptions WeightSerializerOptions = new()
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
        => _layers.Forward(input, inference);

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

    public void SaveWeights(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be empty.", nameof(filePath));

        ModelWeightsDto dto = BuildWeightsDto();

        string? directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string json = JsonSerializer.Serialize(dto, WeightSerializerOptions);
        File.WriteAllText(filePath, json);
    }

    public void LoadWeights(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be empty.", nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException("Weights file was not found.", filePath);

        string json = File.ReadAllText(filePath);
        ModelWeightsDto? dto = JsonSerializer.Deserialize<ModelWeightsDto>(json, WeightSerializerOptions);
        if (dto is null)
            throw new InvalidOperationException("Unable to deserialize weights file.");

        ApplyWeights(dto);
    }

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

    public int GetParamCount()
        => _layers.Sum(l => (int?)l.GetParamCount()) ?? 0;

    #endregion Checkpoint

    private ModelWeightsDto BuildWeightsDto()
    {
        List<LayerWeightsDto> layers = new List<LayerWeightsDto>(_layers.Count);
        foreach (Layer layer in _layers)
        {
            layers.Add(SerializeLayer(layer));
        }

        return new ModelWeightsDto(CurrentWeightsFormatVersion, layers);
    }

    private LayerWeightsDto SerializeLayer(Layer layer)
    {
        IReadOnlyList<Operation> operations = layer.GetOperations();
        List<OperationWeightsDto> serializedOperations = new();

        foreach (Operation operation in operations)
        {
            if (operation is not IParameterStateProvider provider)
                continue;

            ParameterSnapshot snapshot = provider.Capture();
            serializedOperations.Add(new OperationWeightsDto(GetTypeIdentifier(operation.GetType()), ParameterDataDto.FromSnapshot(snapshot)));
        }

        return new LayerWeightsDto(GetTypeIdentifier(layer.GetType()), serializedOperations);
    }

    private void ApplyWeights(ModelWeightsDto dto)
    {
        if (dto.Version != CurrentWeightsFormatVersion)
            throw new InvalidOperationException($"Unsupported weights file version '{dto.Version}'. Expected '{CurrentWeightsFormatVersion}'.");

        if (dto.Layers.Count != _layers.Count)
            throw new InvalidOperationException($"Layer count mismatch. Model has {_layers.Count} layers but weights file has {dto.Layers.Count}.");

        for (int layerIndex = 0; layerIndex < _layers.Count; layerIndex++)
        {
            Layer layer = _layers[layerIndex];
            LayerWeightsDto layerDto = dto.Layers[layerIndex];

            EnsureTypeMatch(layerDto.LayerType, layer.GetType(), layerIndex, operationIndex: null);

            IReadOnlyList<Operation> operations = layer.GetOperations();
            List<Operation> parameterizedOperations = operations.Where(op => op is IParameterStateProvider).ToList();

            if (parameterizedOperations.Count != layerDto.Operations.Count)
            {
                throw new InvalidOperationException($"Operation count mismatch for layer '{layer.GetType().Name}' at index {layerIndex}. Expected {parameterizedOperations.Count} operations with parameters but file has {layerDto.Operations.Count}.");
            }

            for (int opIndex = 0; opIndex < parameterizedOperations.Count; opIndex++)
            {
                Operation operation = parameterizedOperations[opIndex];
                OperationWeightsDto operationDto = layerDto.Operations[opIndex];

                EnsureTypeMatch(operationDto.OperationType, operation.GetType(), layerIndex, opIndex);

                IParameterStateProvider provider = (IParameterStateProvider)operation;
                provider.Restore(operationDto.Parameter.ToSnapshot());
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

    private static string GetTypeIdentifier(Type type)
        => type.AssemblyQualifiedName ?? type.FullName ?? type.Name;

    public List<string> Describe(int indentation)
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

    private sealed record ModelWeightsDto(int Version, List<LayerWeightsDto> Layers);

    private sealed record LayerWeightsDto(string LayerType, List<OperationWeightsDto> Operations);

    private sealed record OperationWeightsDto(string OperationType, ParameterDataDto Parameter);

    private sealed record ParameterDataDto(int[] Shape, float[] Values)
    {
        public ParameterSnapshot ToSnapshot()
        {
            int[] shapeCopy = new int[Shape.Length];
            Array.Copy(Shape, shapeCopy, Shape.Length);
            float[] valueCopy = new float[Values.Length];
            Array.Copy(Values, valueCopy, Values.Length);
            return new ParameterSnapshot(shapeCopy, valueCopy);
        }

        public static ParameterDataDto FromSnapshot(ParameterSnapshot snapshot)
        {
            int[] shapeCopy = new int[snapshot.Shape.Length];
            Array.Copy(snapshot.Shape, shapeCopy, shapeCopy.Length);
            float[] valueCopy = new float[snapshot.Values.Length];
            Array.Copy(snapshot.Values, valueCopy, valueCopy.Length);
            return new ParameterDataDto(shapeCopy, valueCopy);
        }
    }
}
