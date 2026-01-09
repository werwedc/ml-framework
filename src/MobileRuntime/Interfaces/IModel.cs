using System;
using System.Threading.Tasks;

namespace MLFramework.MobileRuntime
{
    /// <summary>
    /// Interface for a loaded model in the mobile runtime.
    /// </summary>
    public interface IModel : IDisposable
    {
        /// <summary>
        /// Name of the model.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Array of input tensor information.
        /// </summary>
        InputInfo[] Inputs { get; }

        /// <summary>
        /// Array of output tensor information.
        /// </summary>
        OutputInfo[] Outputs { get; }

        /// <summary>
        /// Performs synchronous inference on the model.
        /// </summary>
        /// <param name="inputs">Input tensors.</param>
        /// <returns>Output tensors.</returns>
        ITensor[] Predict(ITensor[] inputs);

        /// <summary>
        /// Performs asynchronous inference on the model.
        /// </summary>
        /// <param name="inputs">Input tensors.</param>
        /// <returns>Output tensors.</returns>
        Task<ITensor[]> PredictAsync(ITensor[] inputs);

        /// <summary>
        /// Gets the memory footprint of the model in bytes.
        /// </summary>
        long MemoryFootprint { get; }

        /// <summary>
        /// Gets the format of the model.
        /// </summary>
        ModelFormat Format { get; }
    }
}
