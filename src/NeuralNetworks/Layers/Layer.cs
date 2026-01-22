// Neural Networks in C♯
// File name: Layer.cs
// www.kowaliszyn.pl, 2025

using System.Diagnostics;
using System.Reflection.Emit;

using NeuralNetworks.Core.Operations;
using NeuralNetworks.Layers.Dtos;
using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations;
using NeuralNetworks.Operations.Interfaces;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Utils.ModelUtils;

namespace NeuralNetworks.Layers;

public abstract class Layer
{
    bool _registered = false;

    public void SetRegistered()
    {
        if (_registered)
            throw new InvalidOperationException("Layer is already registered.");
        _registered = true;
    }

    public abstract Type GetOutputType();
    public abstract Type GetInputType();
    public abstract object Forward(object input, bool inference);
    public abstract object Backward(object outputGradient);
    public abstract void UpdateParams(Optimizer optimizer);
    public abstract int GetParamCount();
    internal abstract LayerParams GetParams();

    //internal virtual IReadOnlyList<Operation> GetOperations()
    //    => throw new InvalidOperationException($"Layer '{GetType().Name}' does not expose its operations.");

    internal abstract void ApplyParams(LayerParams layerParams, int layerIndex);

    internal virtual bool IsInitialized => true;
}

public abstract class Layer<TIn, TOut> : Layer
    where TIn : notnull
    where TOut : notnull
{
    private TOut? _output;
    private TIn? _input;

    private OperationList<TIn, TOut>? _operations;

    protected TIn? Input => _input;

    /// <summary>
    /// Passes input forward through a series of operations.
    /// </summary>
    /// <param name="input">Input matrix.</param>
    /// <returns>Output matrix.</returns>
    public TOut Forward(TIn input, bool inference)
    {
        bool firstPass = _input is null;

        // We store the pointer to the input array so we can check the shape of the input gradient in the backward pass.
        // We also use this input (its dimensions) in the SetupLayer method.
        _input = input;
        if (firstPass)
        {
            // First pass, set up the layer.
            SetupLayer();
        }

        Debug.Assert(_operations != null, "Operations were not set up.");

        // As above, we store the pointer to the output array so we can check the shape of the output gradient in the backward pass.
        _output = _operations.Forward(input, inference);

        return _output;
    }

    /// <summary>
    /// Passes <paramref name="outputGradient"/> backward through a series of operations.
    /// </summary>
    /// <remarks>
    /// Checks appropriate shapes. 
    /// </remarks>
    public TIn Backward(TOut outputGradient)
    {
        EnsureSameShapeForOutput(_output, outputGradient);

        Debug.Assert(_operations != null, "Operations were not set up.");

        TIn inputGradient = _operations.Backward(outputGradient);

        //_paramGradients = Operations
        //    .OfType<ParamOperation>()
        //    .Select(po => po.ParamGradient)
        //    .ToList();

        EnsureSameShapeForInput(_input, inputGradient);

        return inputGradient;
    }

    public override void UpdateParams(Optimizer optimizer)
    {
        Debug.Assert(_operations != null, "Operations were not set up.");

        _operations.UpdateParams(this, optimizer);
    }

    public abstract OperationListBuilder<TIn, TOut> CreateOperationListBuilder();

    protected virtual void SetupLayer()
        // Build the operation list
        => _operations = CreateOperationListBuilder().Build();

    protected static OperationListBuilder<TIn, TOpOut> AddOperation<TOpOut>(Operation<TIn, TOpOut> operation)
        where TOpOut : notnull
        => new(operation);

    public override Type GetOutputType() => typeof(TOut);

    public override Type GetInputType() => typeof(TIn);

    public override object Forward(object input, bool inference) => Forward((TIn)input, inference);

    public override object Backward(object outputGradient) => Backward((TOut)outputGradient);

    [Conditional("DEBUG")]
    protected abstract void EnsureSameShapeForInput(TIn? input, TIn? inputGradient);

    [Conditional("DEBUG")]
    protected abstract void EnsureSameShapeForOutput(TOut? output, TOut? outputGradient);

    public override int GetParamCount()
    {
        Debug.Assert(_operations != null, "Operations were not set up.");

        return _operations.GetParamCount();
    }

    //internal override IReadOnlyList<Operation> GetOperations()
    //{
    //    if (_operations is null)
    //        throw new InvalidOperationException($"Layer '{GetType().Name}' is not initialized. Run a forward pass before accessing operations.");

    //    return _operations;
    //}

    internal override bool IsInitialized => _operations is not null;

    internal override LayerParams GetParams()
    {
        Debug.Assert(_operations != null, "Operations were not set up.");

        List<OperationSerializationDto> serializedOperations = [];

        foreach (Operation operation in _operations)
        {
            if (operation is not IParamOperation paramOperation)
                continue;

            ParameterSnapshot snapshot = paramOperation.GetSnapshot();

            string operationType = GetTypeIdentifier(operation.GetType());
            OperationSerializationDto serializedOperation = new(operationType, ParameterDataDto.FromSnapshot(snapshot));
            serializedOperations.Add(serializedOperation);
        }

        string layerType = GetTypeIdentifier(GetType());
        return new LayerParams(layerType, serializedOperations);
    }

    override internal void ApplyParams(LayerParams layerParams, int layerIndex)
    {
        Debug.Assert(_operations != null, "Operations were not set up.");

        EnsureTypeMatch(layerParams.LayerType, GetType(), layerIndex);

        List<Operation> parameterizedOperations = _operations.Where(op => op is IParamOperation).ToList();

        int operationCount = parameterizedOperations.Count;
        int serializedOperationCount = layerParams.Operations.Count;

        if (operationCount != serializedOperationCount)
            throw new InvalidOperationException($"Operation count mismatch. Layer '{GetType().Name}' at index {layerIndex} has {operationCount} operations but {serializedOperationCount} were found in the serialized data.");

        for (int operationIndex = 0; operationIndex < parameterizedOperations.Count; operationIndex++)
        {
            Operation operation = parameterizedOperations[operationIndex];
            OperationSerializationDto operationDto = layerParams.Operations[operationIndex];

            EnsureTypeMatch(operationDto.OperationType, operation.GetType(), layerIndex, operationIndex);

            IParamOperation provider = (IParamOperation)operation;
            provider.Restore(operationDto.Parameters.ToSnapshot());
        }

        //for (int i = 0; i < operationCount; i++)
        //{
        //    Operation operation = _operations[i];
        //    if (operation is not IParamOperation paramOperation)
        //        continue;
        //    OperationSerializationDto serializedOperation = layerParams.Operations[i];
        //    EnsureTypeMatch(serializedOperation.OperationType, operation.GetType(), layerIndex: -1, operationIndex: i);
        //    ParameterDataDto parameterDataDto = serializedOperation.ParameterData;
        //    ParameterSnapshot snapshot = parameterDataDto.ToSnapshot();
        //    paramOperation.ApplySnapshot(snapshot);
        //}
    }

}
