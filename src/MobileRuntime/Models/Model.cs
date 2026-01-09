using System;
using System.Threading.Tasks;

namespace MLFramework.MobileRuntime
{
    /// <summary>
    /// Abstract base class for model implementations.
    /// </summary>
    public abstract class Model : IModel
    {
        /// <summary>
        /// Gets the name of the model.
        /// </summary>
        public abstract string Name { get; }

        /// <summary>
        /// Gets the array of input tensor information.
        /// </summary>
        public abstract InputInfo[] Inputs { get; }

        /// <summary>
        /// Gets the array of output tensor information.
        /// </summary>
        public abstract OutputInfo[] Outputs { get; }

        /// <summary>
        /// Gets the memory footprint of the model in bytes.
        /// </summary>
        public abstract long MemoryFootprint { get; }

        /// <summary>
        /// Gets the format of the model.
        /// </summary>
        public abstract ModelFormat Format { get; }

        /// <summary>
        /// Performs synchronous inference on the model.
        /// </summary>
        /// <param name="inputs">Input tensors.</param>
        /// <returns>Output tensors.</returns>
        public abstract ITensor[] Predict(ITensor[] inputs);

        /// <summary>
        /// Performs asynchronous inference on the model.
        /// </summary>
        /// <param name="inputs">Input tensors.</param>
        /// <returns>Output tensors.</returns>
        public abstract Task<ITensor[]> PredictAsync(ITensor[] inputs);

        /// <summary>
        /// Releases resources used by the model.
        /// </summary>
        public abstract void Dispose();

        /// <summary>
        /// Validates that input tensors match the expected input specifications.
        /// </summary>
        /// <param name="inputs">Input tensors to validate.</param>
        /// <exception cref="ArgumentException">Thrown if inputs are invalid.</exception>
        protected virtual void ValidateInputs(ITensor[] inputs)
        {
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));

            if (inputs.Length != Inputs.Length)
                throw new ArgumentException($"Expected {Inputs.Length} inputs, but got {inputs.Length}", nameof(inputs));

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] == null)
                    throw new ArgumentException($"Input tensor at index {i} is null", nameof(inputs));

                if (inputs[i].DataType != Inputs[i].DataType)
                    throw new ArgumentException(
                        $"Input tensor at index {i} has data type {inputs[i].DataType}, expected {Inputs[i].DataType}",
                        nameof(inputs));
            }
        }
    }
}
