// Machine Learning Utils
// File name: OperationListBuilder.cs
// Code It Yourself with .NET, 2024

using NeuralNetworks.Operations;

namespace NeuralNetworks.Layers.OperationList;

/*
 *  OperationBuilder and OperationBuilder<TIn, TOut> are in the Layers.OperationList namespace, because they are used to build layers.
 */

public abstract class OperationListBuilder(OperationListBuilder? parent = null)
{
    public OperationListBuilder? Parent => parent;

    public Operation Operation { get; protected set; } = null!;
}

public class OperationListBuilder<TIn, TOut> : OperationListBuilder
    where TIn : notnull
    where TOut : notnull
{
    internal OperationListBuilder(Operation<TIn, TOut> operation): base()
    {
        Operation = operation;
    }

    private OperationListBuilder(Operation operation, OperationListBuilder parent) : base(parent)
    {
        Operation = operation;
    }

    public OperationListBuilder<TIn, TNextOut> AddOperation<TNextOut>(Operation<TOut, TNextOut> operation)
        where TNextOut : notnull
        => new(operation, this);

    public OperationList<TIn, TOut> Build()
    {
        // Traverse the builder chain backwards to get all the operations in the reverse order
        OperationList<TIn, TOut> operations = [];

        OperationListBuilder? builder = this;
        while (builder != null)
        {
            builder.Operation.SetRegistered();
            operations.Insert(0, builder.Operation);
            builder = builder.Parent;
        }

        return operations;
    }
}
