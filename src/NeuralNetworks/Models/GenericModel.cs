// Neural Networks in C♯
// File name: GenericModel.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;
using NeuralNetworks.Losses;

namespace NeuralNetworks.Models;

public class GenericModel<TInputData, TPrediction> : Model<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    public GenericModel(LayerListBuilder<TInputData, TPrediction> layerListBuilder, Loss<TPrediction> lossFunction, SeededRandom? random)
        : base(layerListBuilder, lossFunction, random)
    {
    }

    internal protected override LayerListBuilder<TInputData, TPrediction> CreateLayerListBuilderInternal() => throw new InvalidOperationException();
}
