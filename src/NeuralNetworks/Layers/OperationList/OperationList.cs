using NeuralNetworks.Operations;
using NeuralNetworks.Operations.Interfaces;
using NeuralNetworks.Optimizers;

namespace NeuralNetworks.Layers.OperationList;

public class OperationList<TIn, TOut> : List<Operation>
    where TIn : notnull
    where TOut : notnull
{
    public TOut Forward(TIn input, bool inference)
    {
        // I like to call it "stream" because it's like a stream of data flowing through the network.
        object stream = input;
        foreach (Operation operation in this)
        {
            stream = operation.Forward(stream, inference);
        }
        return (TOut)stream;
    }

    public TIn Backward(TOut outputGradient)
    {
        object stream = outputGradient;
        foreach (Operation operation in this.Reverse<Operation>())
        {
            stream = operation.Backward(stream);
        }
        return (TIn)stream;
    }

    /// <param name="layer">The layer this operation list belongs to.</param>
    public void UpdateParams(Layer layer, Optimizer optimizer)
    {
        foreach (IParamOperation operation in this.OfType<IParamOperation>())
        {
            operation.UpdateParams(layer, optimizer);
        }
    }

    public int GetParamCount()
    {
        return this
            .OfType<IParamOperation>()
            .Sum(po => (int?)po.GetParamCount()) ?? 0;
    }
}
