// Neural Networks in C♯
// File name: Layer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations;
using NeuralNetworks.Operations.Parameterized;
using NeuralNetworks.Optimizers;

using static NeuralNetworks.Core.GenericArrayUtils;
using static NeuralNetworks.Utils.ModelUtils;

namespace NeuralNetworks.Layers;

public abstract class Layer
{
    private bool _registered = false;

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
        Layer<TIn, TOut>.EnsureSameShapeForOutput(_output, outputGradient);

        Debug.Assert(_operations != null, "Operations were not set up.");

        TIn inputGradient = _operations.Backward(outputGradient);

        //_paramGradients = Operations
        //    .OfType<ParamOperation>()
        //    .Select(po => po.ParamGradient)
        //    .ToList();

        Layer<TIn, TOut>.EnsureSameShapeForInput(_input, inputGradient);

        return inputGradient;
    }

    public override void UpdateParams(Optimizer optimizer)
    {
        Debug.Assert(_operations != null, "Operations were not set up.");

        _operations.UpdateParams(optimizer);
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
    private static void EnsureSameShapeForInput(TIn? input, TIn? inputGradient)
        => EnsureSameShape(input, inputGradient);

    [Conditional("DEBUG")]
    private static void EnsureSameShapeForOutput(TOut? output, TOut? outputGradient)
        => EnsureSameShape(output, outputGradient);

    public override int GetParamCount()
    {
        Debug.Assert(_operations != null, "Operations were not set up.");

        return _operations.GetParamCount();
    }

    internal override bool IsInitialized => _operations is not null;

    internal override LayerParams GetParams()
    {
        Debug.Assert(_operations != null, "Operations were not set up.");

        List<ParamOperationData> paramOperationDatas = _operations
            .OfType<IParamOperation>()
            .Select(paramOperation => paramOperation.GetData())
            .ToList();

        string layerType = GetTypeIdentifier(GetType());
        return new LayerParams(layerType, paramOperationDatas);
    }

    internal override void ApplyParams(LayerParams layerParams, int layerIndex)
    {
        Debug.Assert(_operations != null, "Operations were not set up.");

        EnsureTypeMatch(layerParams.LayerType, GetType(), layerIndex);

        List<IParamOperation> paramOperations = _operations
            .OfType<IParamOperation>()
            .ToList();

        int operationCount = paramOperations.Count;
        int serializedOperationCount = layerParams.Operations.Count;

        if (operationCount != serializedOperationCount)
            throw new InvalidOperationException($"Operation count mismatch. Layer '{GetType().Name}' at index {layerIndex} has {operationCount} operations but {serializedOperationCount} were found in the serialized data.");

        for (int operationIndex = 0; operationIndex < paramOperations.Count; operationIndex++)
        {
            IParamOperation operation = paramOperations[operationIndex];
            ParamOperationData operationDto = layerParams.Operations[operationIndex];
            operation.ApplyData(operationDto, layerIndex, operationIndex);
        }
    }

}
