// Neural Networks in C♯
// File name: BaseModel.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;
using NeuralNetworks.Layers;
using NeuralNetworks.Losses;
using NeuralNetworks.Models.LayerList;

namespace NeuralNetworks.Models;

public abstract class BaseModel<TInputData, TPrediction> : Model<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    public BaseModel(Loss<TPrediction> lossFunction, SeededRandom? random, string? modelFilePath = null)
        : base(null, lossFunction, random, modelFilePath)
    {
    }

    protected static LayerListBuilder<TInputData, TLayerOut> AddLayer<TLayerOut>(Layer<TInputData, TLayerOut> layer)
        where TLayerOut : notnull
        => new(layer);
}
