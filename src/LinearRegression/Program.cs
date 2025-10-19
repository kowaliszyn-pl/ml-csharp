// Neural Networks in C♯
// File name: Program.cs
// www.kowaliszyn.pl, 2025

Console.OutputEncoding = System.Text.Encoding.UTF8;

// 1. Set the parameters for the model

const float LearningRate = 0.0005f;
const int Iterations = 35_000; // 4
const int PrintEvery = 1_000; // 1

// 2. Prepare training data

float[][] data = [
    [10, 100],
    [20, 80],
    [30, 60],
    [40, 40],
    [50, 20],
];

// 3. Initialize model

float a = 0, b = 0;

// 4. Training loop

for (int iteration = 1; iteration <= Iterations; iteration++)
{
    // Initialize accumulators for errors
    float sumErrorValue = 0, sumError = 0, squaredError = 0;

    foreach (float[] sample in data)
    {
        float x = sample[0];
        float y = sample[1];

        // Prediction and error calculation
        float prediction = a * x + b;
        float error = y - prediction;

        // Accumulate squared error and gradients
        squaredError += error * error;
        sumErrorValue += error * x;
        sumError += error;
    }

    // Number of samples
    int n = data.Length;

    // MSE
    float meanSquaredError = squaredError / n;

    // Calculate gradients (partial derivatives)
    float deltaA = -2.0f / n * sumErrorValue;
    float deltaB = -2.0f / n * sumError;

    // Update regression parameters
    a -= LearningRate * deltaA;
    b -= LearningRate * deltaB;

    if (iteration % PrintEvery == 0)
        Console.WriteLine($"Iteration: {iteration,5}, MSE: {meanSquaredError,10:F5}, ∂MSE/∂a: {deltaA,10:F4}, ∂MSE/∂b: {deltaB,10:F4}, a: {a,9:F4}, b: {b,9:F4}");
}

// 5. Output learned parameters

Console.WriteLine();
Console.WriteLine($"Learned parameters:");
Console.WriteLine($"a = {a:F4}, b = {b:F4}");
Console.WriteLine($"Expected parameters:");
Console.WriteLine($"a = {-2:F4}, b = {120:F4}");
Console.ReadLine();