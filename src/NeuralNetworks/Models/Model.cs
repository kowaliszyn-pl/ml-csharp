// Neural Networks in C♯
// File name: Model.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Models;

//public abstract class Model
//{
//    public abstract void UpdateParams(Optimizer optimizer);
//}

//public abstract class ModelBuilder<TInputData, TPrediction>
//    where TInputData : notnull
//    where TPrediction : notnull
//{
//    private readonly LayerListBuilder<TInputData, TPrediction>? _layerListBuilder;

//    public ModelBuilder(LayerListBuilder<TInputData, TPrediction>? layerListBuilder)
//    {
//        _layerListBuilder = layerListBuilder;
//    }

//    protected virtual LayerListBuilder<TInputData, TPrediction> GetLayerListBuilder()
//    {
//        return _layerListBuilder ?? CreateListBuilder();
//    }
//}

public abstract class BaseModel<TInputData, TPrediction> //: Model
    where TInputData : notnull
    where TPrediction : notnull
{
    private LayerList<TInputData, TPrediction> _layers;
    private Loss<TPrediction> _lossFunction;
    private float _lastLoss;
    private readonly LayerListBuilder<TInputData, TPrediction>? _layerListBuilder;

    protected BaseModel(LayerListBuilder<TInputData, TPrediction>? layerListBuilder, Loss<TPrediction> lossFunction, SeededRandom? random)
    {
        _lossFunction = lossFunction;
        Random = random;
        // _layers = CreateLayerListBuilder().Build();
        // _layers = GetLayers();
        // _layers = layerListBuilder.Build();
        _layerListBuilder = layerListBuilder;
        _layers = BuildLayers();
    }

    protected virtual LayerList<TInputData, TPrediction> BuildLayers()
    {
        LayerListBuilder<TInputData, TPrediction> layerListBuilder = _layerListBuilder ?? CreateLayerListBuilder();
        return layerListBuilder.Build();
    }

    public IReadOnlyList<Layer> Layers => _layers;

    public Loss<TPrediction> LossFunction => _lossFunction;

    protected SeededRandom? Random { get; }

    //protected abstract LayerList<TInputData, TPrediction> GetLayers();

    protected abstract LayerListBuilder<TInputData, TPrediction> CreateLayerListBuilder();

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
        _lossFunction = _checkpoint._lossFunction;
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
        clone._lossFunction = _lossFunction.Clone();
        return clone;
    }

    public int GetParamCount()
        => _layers.Sum(l => (int?)l.GetParamCount()) ?? 0;

    #endregion Checkpoint

}

public class Model<TInputData, TPrediction>: BaseModel<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    public Model(LayerListBuilder<TInputData, TPrediction> layerListBuilder, Loss<TPrediction> lossFunction, SeededRandom? random)
        : base(layerListBuilder, lossFunction, random)
    {
    }

    protected override LayerListBuilder<TInputData, TPrediction> CreateLayerListBuilder() => throw new InvalidOperationException();

    // protected override LayerList<TInputData, TPrediction> GetLayers() => _layerList;
}

public abstract class CustomModel<TInputData, TPrediction> : BaseModel<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    public CustomModel(Loss<TPrediction> lossFunction, SeededRandom? random)
        : base(null, lossFunction, random)
    {
    }

    protected static LayerListBuilder<TInputData, TLayerOut> AddLayer<TLayerOut>(Layer<TInputData, TLayerOut> layer)
        where TLayerOut : notnull
        => new(layer);
}

/*
public class Model(List<Layer> layers, Loss lossFunction)
{
    private List<Layer> _layers = layers;
    private Loss _lossFunction = lossFunction;
    private float _lastLoss;

    public IReadOnlyList<Layer> Layers => _layers;

    public Loss LossFunction => _lossFunction;

    public float LastLoss => _lastLoss;

    //public int ParameterCount => _layers
    //    .SelectMany(layer => layer.Params)
    //    .Sum(paramMatrix => paramMatrix.Array.Length);

    /// <summary>
    /// Performs the forward pass of the neural network on the given batch of input data.
    /// </summary>
    /// <param name="batch">The input data batch.</param>
    /// <param name="inference">A flag indicating whether the forward pass is for inference or training.</param>
    /// <returns>The output of the neural network.</returns>
    public Matrix Forward(Matrix batch, bool inference)
    {
        Matrix input = batch;
        foreach (Layer layer in _layers)
        {
            input = layer.Forward(input, inference);
        }
        return input;
    }

    public void Backward(Matrix lossGrad)
    {
        Matrix grad = lossGrad;
        foreach (Layer layer in _layers.Reverse<Layer>())
        {
            grad = layer.Backward(grad);
        }
    }

    public float TrainBatch(Matrix xBatch, Matrix yBatch)
    {
        Matrix predictions = Forward(xBatch, false);
        _lastLoss = _lossFunction.Forward(predictions, yBatch);
        Backward(_lossFunction.Backward());
        return _lastLoss;
    }

    public Matrix[] GetAllParams() => _layers.SelectMany(layer => layer.Params).ToArray();

    internal Matrix[] GetAllParamGradients() => _layers.SelectMany(layer => layer.ParamGradients).ToArray();


    private Model? _checkpoint;

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
    public Model Clone()
    {
        Model clone = (Model)MemberwiseClone();
        clone._layers = _layers.Select(l => l.Clone()).ToList();
        clone._lossFunction = _lossFunction.Clone();
        return clone;
    }
}

*/