// Neural Networks in C♯
// File name: OperationList.cs
// www.kowaliszyn.pl, 2025 - 2026

using NeuralNetworks.Operations;

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
}
