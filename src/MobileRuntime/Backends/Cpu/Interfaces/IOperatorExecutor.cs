using MLFramework.MobileRuntime.Backends.Cpu.Models;

namespace MLFramework.MobileRuntime.Backends.Cpu.Interfaces
{
    using System.Collections.Generic;

    /// <summary>
    /// Interface for operator execution.
    /// </summary>
    public interface IOperatorExecutor
    {
        /// <summary>
        /// Type of operator this executor handles.
        /// </summary>
        OperatorType OperatorType { get; }

        /// <summary>
        /// Executes the operator.
        /// </summary>
        /// <param name="inputs">Input tensors.</param>
        /// <param name="parameters">Operator parameters.</param>
        /// <returns>Output tensor.</returns>
        ITensor Execute(ITensor[] inputs, Dictionary<string, object> parameters);

        /// <summary>
        /// Checks if this operator can be fused with another.
        /// </summary>
        /// <param name="other">The other operator executor.</param>
        /// <returns>True if fusion is supported.</returns>
        bool CanFuseWith(IOperatorExecutor other);

        /// <summary>
        /// Executes a fused operation.
        /// </summary>
        /// <param name="executors">Array of executors in the fused chain.</param>
        /// <param name="inputs">Input arrays for each executor.</param>
        /// <param name="parameters">Parameters for each executor.</param>
        /// <returns>Output tensor.</returns>
        ITensor ExecuteFused(IOperatorExecutor[] executors, ITensor[][] inputs, Dictionary<string, object>[] parameters);
    }
}
