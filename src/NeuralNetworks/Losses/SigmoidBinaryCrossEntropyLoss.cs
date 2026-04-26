// Neural Networks in C♯
// File name: BinaryCrossEntropyLoss.cs
// www.kowaliszyn.pl, 2025 - 2026

using static NeuralNetworks.Core.Operations.OperationBackend;

namespace NeuralNetworks.Losses;

public class SigmoidBinaryCrossEntropyLoss(float eps = 1e-7f) : Loss<float[,]>
{
    protected override float CalculateLoss()
        => BinaryCrossEntropyLoss(Prediction, Target); // TODO: calculate Sigmoid inside the loss function just like Softmax in SoftmaxCrossEntropyLoss, to make the gradient easier to calculate and more numerically stable. The current implementation assumes that the Prediction is already passed through a Sigmoid activation function. Apply Sigmoid to logits in the eval function (like the Argmax in MNIST evaluation).

    protected override float[,] CalculateLossGradient()
        => BinaryCrossEntropyLossGradient(Prediction, Target);

    public override string ToString() => $"BinaryCrossEntropyLoss (eps={eps})";
}
