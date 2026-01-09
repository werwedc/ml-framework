using MLFramework.MobileRuntime.Backends.Cpu.Models;

namespace MLFramework.MobileRuntime.Backends.Cpu.Interfaces
{
    using System.Collections.Generic;

    /// <summary>
    /// Interface for CPU execution backend.
    /// </summary>
    public interface ICpuBackend
    {
        /// <summary>
        /// Name of the backend.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Capabilities of the CPU backend.
        /// </summary>
        BackendCapabilities Capabilities { get; }

        /// <summary>
        /// Executes a single operator.
        /// </summary>
        /// <param name="op">Operator descriptor.</param>
        /// <param name="inputs">Input tensors.</param>
        /// <param name="parameters">Operator parameters.</param>
        /// <returns>Output tensor.</returns>
        ITensor Execute(OperatorDescriptor op, ITensor[] inputs, Dictionary<string, object> parameters);

        /// <summary>
        /// Executes a batch of operators efficiently.
        /// </summary>
        /// <param name="ops">Array of operator descriptors.</param>
        /// <param name="tensorRegistry">Registry of tensors by ID.</param>
        /// <returns>Array of output tensors.</returns>
        ITensor[] ExecuteBatch(OperatorDescriptor[] ops, Dictionary<uint, ITensor> tensorRegistry);

        /// <summary>
        /// Gets information about the CPU.
        /// </summary>
        /// <returns>CPU information.</returns>
        CpuInfo GetCpuInfo();

        /// <summary>
        /// Enables or disables vectorization optimizations.
        /// </summary>
        /// <param name="enable">True to enable vectorization.</param>
        void EnableVectorization(bool enable);

        /// <summary>
        /// Enables or disables multi-threading.
        /// </summary>
        /// <param name="enable">True to enable multi-threading.</param>
        /// <param name="maxThreads">Maximum number of threads (0 = auto-detect).</param>
        void EnableMultiThreading(bool enable, int maxThreads = 0);
    }
}
