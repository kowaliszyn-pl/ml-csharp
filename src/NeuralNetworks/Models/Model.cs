// Neural Networks in C♯
// File name: GenericModel.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Models;

public abstract class Model<TInputData, TPrediction> 
    where TInputData : notnull
    where TPrediction : notnull
{
    private LayerList<TInputData, TPrediction> _layers;
    private Loss<TPrediction> _lossFunction;
    private float _lastLoss;

    protected Model(LayerListBuilder<TInputData, TPrediction>? layerListBuilder, Loss<TPrediction> lossFunction, SeededRandom? random)
    {
        _lossFunction = lossFunction;
        Random = random;
        _layers = (layerListBuilder ?? CreateLayerListBuilderInternal()).Build();
    }

    public IReadOnlyList<Layer> Layers => _layers;

    public Loss<TPrediction> LossFunction => _lossFunction;

    protected SeededRandom? Random { get; }

    internal protected abstract LayerListBuilder<TInputData, TPrediction> CreateLayerListBuilderInternal();

    public TPrediction Forward(TInputData input, bool inference)
    {
        return _layers.Forward(input, inference);
    }

    public void Backward(TPrediction lossGrad)
    {
        _layers.Backward(lossGrad);
    }

    public float TrainBatch(TInputData xBatch, TPrediction yBatch)
    {
        TPrediction predictions = Forward(xBatch, false);
        _lastLoss = _lossFunction.Forward(predictions, yBatch);
        Backward(_lossFunction.Backward());
        return _lastLoss;
    }

    public void UpdateParams(Optimizer optimizer)
    {
        _layers.UpdateParams(optimizer);
    }

    #region Checkpoint

    private GenericModel<TInputData, TPrediction>? _checkpoint;

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
        _lossFunction = _checkpoint._lossFunction;
        _lastLoss = _checkpoint._lastLoss;
    }

    /// <summary>
    /// Makes a deep copy of this neural network.
    /// </summary>
    /// <returns></returns>
    public GenericModel<TInputData, TPrediction> Clone()
    {
        GenericModel<TInputData, TPrediction> clone = (GenericModel<TInputData, TPrediction>)MemberwiseClone();
        clone._layers = _layers.Clone();
        clone._lossFunction = _lossFunction.Clone();
        return clone;
    }

    public int GetParamCount()
        => _layers.Sum(l => (int?)l.GetParamCount()) ?? 0;

    #endregion Checkpoint

}
