// Neural Networks in C♯
// File name: Layer.cs
// www.kowaliszyn.pl, 2025 - 2026

using System.Diagnostics;

using NeuralNetworks.Layers.OperationList;
using NeuralNetworks.Operations;
using NeuralNetworks.Operations.Parameterized;
using NeuralNetworks.Optimizers;
using NeuralNetworks.Regularization;

using static NeuralNetworks.Core.GenericArrayRoutines;
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
    public abstract float GetRegularizationLoss();
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

    private IParamOperation[]? _cachedParamOperations;

    protected TIn? Input => _input;

    /// <summary>
    /// Gets the output of the layer after a forward pass.
    /// </summary>
    /// <remarks>
    /// This value can also be used when retrieving an output from a given layer (for example, latent layer in an autoencoder). 
    /// </remarks>
    public TOut? Output => _output;

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
        // This value can also be used when retrieving an output from a given layer (for example, latent layer in an autoencoder).
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
        foreach (IParamOperation operation in GetParamOperations())
        {
            operation.UpdateParams(optimizer);
        }
    }

    public abstract OperationListBuilder<TIn, TOut> CreateOperationListBuilder();

    protected virtual void SetupLayer()
    {
        // Build the operation list
        _operations = CreateOperationListBuilder().Build();

        // Invalidate cached parameter operations since the operation list has changed
        _cachedParamOperations = null;
    }

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
        int paramCount = 0;
        foreach (IParamOperation paramOperation in GetParamOperations())
        {
            paramCount += paramOperation.GetParamCount();
        }

        return paramCount;
    }

    private IParamOperation[] GetParamOperations()
    {
        if (_cachedParamOperations == null)
        {
            Debug.Assert(_operations != null, "Operations were not set up.");

            _cachedParamOperations = [.. _operations.OfType<IParamOperation>()];
        }

        return _cachedParamOperations;
    }

    /// <summary>
    /// Calculates the total regularization loss across all parameter operations.
    /// </summary>
    /// <remarks>Returns 0 when no parameter operations are available. When the <see cref="Regularizer.ReportPenaltyInLoss"/> property of a parameter operation is false, its regularization loss is not included in the total.</remarks>
    /// <returns>The sum of regularization loss values returned by each parameter operation.</returns>
    public override float GetRegularizationLoss()
    {
        float regularizationLoss = 0f;
        foreach (IParamOperation paramOperation in GetParamOperations())
        {
            regularizationLoss += paramOperation.GetRegularizationLoss();
        }
        return regularizationLoss;
    }

    internal override bool IsInitialized => _operations is not null;

    internal override LayerParams GetParams()
    {
        List<ParamOperationData> paramOperationDatas = [.. GetParamOperations()
            .Select(paramOperation => paramOperation.GetData())
        ];

        string layerType = GetTypeIdentifier(GetType());
        return new LayerParams(layerType, paramOperationDatas);
    }

    internal override void ApplyParams(LayerParams layerParams, int layerIndex)
    {
        EnsureTypeMatch(layerParams.LayerType, GetType(), layerIndex);

        IParamOperation[] paramOperations = GetParamOperations();

        int operationCount = paramOperations.Length;
        int serializedOperationCount = layerParams.Operations.Count;

        if (operationCount != serializedOperationCount)
            throw new InvalidOperationException($"Operation count mismatch. Layer '{GetType().Name}' at index {layerIndex} has {operationCount} operations but {serializedOperationCount} were found in the serialized data.");

        for (int operationIndex = 0; operationIndex < paramOperations.Length; operationIndex++)
        {
            IParamOperation operation = paramOperations[operationIndex];
            ParamOperationData operationDto = layerParams.Operations[operationIndex];
            operation.ApplyData(operationDto, layerIndex, operationIndex);
        }
    }

}
