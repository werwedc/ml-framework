using System;

namespace MLFramework.IR.Backend
{
    /// <summary>
    /// Compiler for using backends to compile HLIR modules
    /// </summary>
    public class BackendCompiler
    {
        /// <summary>
        /// Compile a module using the specified backend
        /// </summary>
        public static CompilationResult Compile(HLIRModule module, string backendName,
                                               CompilationOptions options = null)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));

            if (string.IsNullOrEmpty(backendName))
                throw new ArgumentException("Backend name cannot be null or empty", nameof(backendName));

            options = options ?? new CompilationOptions();

            var backend = BackendRegistry.Instance.GetBackend(backendName);
            if (backend == null)
                throw new ArgumentException($"Backend '{backendName}' not found");

            if (!backend.CanCompile(module))
                throw new InvalidOperationException($"Backend '{backendName}' cannot compile this module");

            var result = new CompilationResult();

            try
            {
                if (options.Verbose)
                    Console.WriteLine($"Compiling to {backend.Name} backend...");

                // Lower to backend IR
                if (options.Verbose)
                    Console.WriteLine("Lowering to backend IR...");
                var loweredModule = backend.LowerToBackendIR(module);

                // Optimize
                if (options.Verbose)
                    Console.WriteLine($"Optimizing at {options.OptimizationLevel} level...");
                backend.Optimize(loweredModule, options.OptimizationLevel);

                // Generate code
                if (options.Verbose)
                    Console.WriteLine("Generating code...");
                result.Code = backend.GenerateCode(loweredModule);

                // Generate binary (optional)
                try
                {
                    if (options.Verbose)
                        Console.WriteLine("Generating binary...");
                    result.Binary = backend.GenerateBinary(loweredModule);
                }
                catch (NotImplementedException)
                {
                    // Binary generation is optional
                    if (options.Verbose)
                        Console.WriteLine("Binary generation not implemented (skipping)");
                }

                if (options.Verbose)
                    Console.WriteLine("Compilation successful!");
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.ErrorMessage = ex.Message;
            }

            return result;
        }
    }
}
