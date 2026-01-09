using System;
using System.Threading.Tasks;

namespace MobileRuntime
{
    /// <summary>
    /// Interface for ML model
    /// </summary>
    public interface IModel : IDisposable
    {
        /// <summary>
        /// Gets the model name
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets input tensor information
        /// </summary>
        InputInfo[] Inputs { get; }

        /// <summary>
        /// Gets output tensor information
        /// </summary>
        OutputInfo[] Outputs { get; }

        /// <summary>
        /// Gets the memory footprint in bytes
        /// </summary>
        long MemoryFootprint { get; }

        /// <summary>
        /// Gets the model format
        /// </summary>
        ModelFormat Format { get; }

        /// <summary>
        /// Performs synchronous inference
        /// </summary>
        ITensor[] Predict(ITensor[] inputs);

        /// <summary>
        /// Performs asynchronous inference
        /// </summary>
        Task<ITensor[]> PredictAsync(ITensor[] inputs);
    }
}
