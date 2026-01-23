// Neural Networks in C♯
// File name: GenericModel.cs
// www.kowaliszyn.pl, 2025

using NeuralNetworks.Core;
using NeuralNetworks.Losses;
using NeuralNetworks.Models.LayerList;

namespace NeuralNetworks.Models;

/// <summary>
/// Represents a generic machine learning model that operates on input data of type <typeparamref name="TInputData"/>
/// and produces predictions of type <typeparamref name="TPrediction"/>.
/// </summary>
/// <remarks>
/// This class provides a flexible base for building models with customizable input and prediction types.
/// It is typically used in scenarios where the data and prediction formats are determined by the application domain.
/// The layer list builder and loss function are provided during construction to define the model's architecture. 
/// This class is generally not intended to be subclassed.
/// <example>
/// Example of creating a generic model for float[,] input and output:
/// <code>
/// model = new GenericModel&lt;float[,], float[,]&gt;(
///        layerListBuilder: LayerListBuilder&lt;float[,], float[,]&gt;
///                 .AddLayer(new DenseLayer(4, new Sigmoid(), new GlorotInitializer(commonRandom)))
///                 .AddLayer(new DenseLayer(1, new Linear(), new GlorotInitializer(commonRandom))),
///        lossFunction: new MeanSquaredError(),
///        random: commonRandom);
/// </code>
/// </example>
/// </remarks>
/// <typeparam name="TInputData">The type of input data processed by the model. Must not be null.</typeparam>
/// <typeparam name="TPrediction">The type of prediction output produced by the model. Must not be null.</typeparam>
public class GenericModel<TInputData, TPrediction> : Model<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    /// <summary>
    /// Initializes a new instance of the GenericModel class with the specified layer configuration, loss function, and
    /// optional random seed.
    /// </summary>
    /// <remarks>
    /// <example>
    /// Example of creating a generic model for float[,] input and output:
    /// <code>
    /// model = new GenericModel&lt;float[,], float[,]&gt;(
    ///        layerListBuilder: LayerListBuilder&lt;float[,], float[,]&gt;
    ///                 .AddLayer(new DenseLayer(4, new Sigmoid(), new GlorotInitializer(commonRandom)))
    ///                 .AddLayer(new DenseLayer(1, new Linear(), new GlorotInitializer(commonRandom))),
    ///        lossFunction: new MeanSquaredError(),
    ///        random: commonRandom);
    /// </code>
    /// </example>
    /// </remarks>
    /// <param name="layerListBuilder">The builder that defines the sequence and configuration of layers for the model. Must not be null.</param>
    /// <param name="lossFunction">The loss function used to evaluate prediction accuracy during training. Must not be null.</param>
    /// <param name="random">An optional seeded random number generator used for reproducible initialization. If null, a default random generator is used.</param>
    public GenericModel(LayerListBuilder<TInputData, TPrediction> layerListBuilder, Loss<TPrediction> lossFunction, SeededRandom? random)
        : base(layerListBuilder, lossFunction, random)
    {
    }

    private protected override LayerListBuilder<TInputData, TPrediction> CreateLayerListBuilderPrivate() 
        => throw new InvalidOperationException("Provide layerListBuilder in the constructor.");
}
