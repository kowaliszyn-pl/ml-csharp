namespace NeuralNetworks.Core
{
    public class NumberWithGrad(float number, List<NumberWithGrad> dependsOn, CreationOp creationOp)
    {
        public float Number { get; } = number;
        public float Grad { get; private set; } = 0;

        // define the Add operator

        public static NumberWithGrad operator +(NumberWithGrad a, NumberWithGrad b)
        {
            // add the two numbers
            float result = a.Number + b.Number;
            // create a new NumberWithGrad object
            NumberWithGrad res = new(result, [a, b], CreationOp.Add);
            return res;
        }

        // define the Multiply operator

        public static NumberWithGrad operator *(NumberWithGrad a, NumberWithGrad b)
        {
            // multiply the two numbers
            float result = a.Number * b.Number;
            // create a new NumberWithGrad object
            NumberWithGrad res = new(result, [a, b], CreationOp.Multiply);
            return res;
        }

        // implicit conversion from float to NumberWithGrad

        public static implicit operator NumberWithGrad(float number)
        {
            return new NumberWithGrad(number, [], CreationOp.None);
        }

        public void Backward()
        {
            Backward(1);
        }

        private void Backward(float gradient)
        {
            Grad += gradient;

            // if the creation operation is an addition
            if (creationOp == CreationOp.Add)
            {
                // the gradient of the first number is the same as the gradient of the result
                dependsOn[0].Backward(gradient);
                // the gradient of the second number is the same as the gradient of the result
                dependsOn[1].Backward(gradient);
            }
            // if the creation operation is a multiplication
            else if (creationOp == CreationOp.Multiply)
            {
                // the gradient of the first number is the gradient of the result multiplied by the second number
                dependsOn[0].Backward(gradient * dependsOn[1].Number);
                // the gradient of the second number is the gradient of the result multiplied by the first number
                dependsOn[1].Backward(gradient * dependsOn[0].Number);
            }
        }
    }
}
