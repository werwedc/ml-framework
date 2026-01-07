using System.Collections.Generic;

namespace MLFramework.IR.Backend
{
    /// <summary>
    /// Interface for compilation backends (CPU, GPU, etc.)
    /// </summary>
    public interface IBackend
    {
        /// <summary>
        /// Name of the backend
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Target triple (e.g., "x86_64-unknown-linux-gnu")
        /// </summary>
        string TargetTriple { get; }

        /// <summary>
        /// Check if the backend can compile this module
        /// </summary>
        bool CanCompile(HLIRModule module);

        /// <summary>
        /// Compile the module with given options
        /// </summary>
        void Compile(HLIRModule module, CompilationOptions options);

        /// <summary>
        /// Lower HLIR to backend-specific IR
        /// </summary>
        HLIRModule LowerToBackendIR(HLIRModule module);

        /// <summary>
        /// Generate source code from the module
        /// </summary>
        string GenerateCode(HLIRModule module);

        /// <summary>
        /// Generate binary from the module
        /// </summary>
        byte[] GenerateBinary(HLIRModule module);

        /// <summary>
        /// Optimize the module
        /// </summary>
        void Optimize(HLIRModule module, OptimizationLevel level);

        /// <summary>
        /// Execute the module (optional, for testing)
        /// </summary>
        void Execute(HLIRModule module, Dictionary<string, object> inputs);
    }
}
