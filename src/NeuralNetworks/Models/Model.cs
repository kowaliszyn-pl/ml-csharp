// Neural Networks in C♯
// File name: Model.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Models.LayerList;
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

}
