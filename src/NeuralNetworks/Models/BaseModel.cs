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
    public BaseModel(Loss<TPrediction>? defaultLossFunction = null, SeededRandom? random = null, string? modelFilePath = null)
        : base(null, defaultLossFunction, random, modelFilePath)
    {
    }

    public BaseModel(SeededRandom random)
       : this(null, random, null)
    {
    }

    /// <summary>
    /// Load model parameters from a file. The file should contain the parameters of the model in a format compatible
    /// with the model's architecture. This constructor allows initializing the model with pre-trained parameters,
    /// enabling tasks such as inference or fine-tuning on new data.
    /// </summary>
    /// <param name="modelFilePath"></param>
    public BaseModel(string? modelFilePath)
       : this(null, null, modelFilePath)
    {
    }

    protected static LayerListBuilder<TInputData, TLayerOut> AddLayer<TLayerOut>(Layer<TInputData, TLayerOut> layer)
        where TLayerOut : notnull
        => new(layer);

    private protected override LayerListBuilder<TInputData, TPrediction> CreateLayerListBuilderPrivate()
        => CreateLayerListBuilder();

    protected abstract LayerListBuilder<TInputData, TPrediction> CreateLayerListBuilder();
}
