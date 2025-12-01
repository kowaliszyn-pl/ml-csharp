// Machine Learning Utils
// File name: LayerList.cs
// Code It Yourself with .NET, 2024


using NeuralNetworks.Layers;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Models;

public class LayerList<TIn, TOut> : List<Layer>
    where TIn : notnull
    where TOut : notnull
{
    public TOut Forward(TIn input, bool inference)
    {
        object stream = input;
        foreach (Layer layer in this)
        {
            stream = layer.Forward(stream, inference);
        }
        return (TOut)stream;
    }

    public void Backward(TOut lossGrad)
    {
        object stream = lossGrad;
        foreach (Layer layer in this.Reverse<Layer>())
        {
            stream = layer.Backward(stream);
        }
    }

    internal void UpdateParams(Optimizer optimizer)
    {
        foreach (Layer layer in this)
        {
            layer.UpdateParams(optimizer);
        }
    }

    internal LayerList<TIn, TOut> Clone() 
        => (LayerList<TIn, TOut>)MemberwiseClone();
}
