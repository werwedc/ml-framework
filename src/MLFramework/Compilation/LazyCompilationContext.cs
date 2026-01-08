using MLFramework.Fusion;
using MLFramework.Fusion.Dynamic;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Compilation;

/// <summary>
/// Context for lazy kernel compilation
/// </summary>
public class LazyCompilationContext
{
    private readonly object _lockObject = new object();

    /// <summary>
    /// Gets the operation to compile
    /// </summary>
    public required Operation Operation { get; init; }

    /// <summary>
    /// Gets the input tensor shapes
    /// </summary>
    public required List<int[]> InputShapes { get; init; }

    /// <summary>
    /// Gets the output tensor shapes
    /// </summary>
    public required List<int[]> OutputShapes { get; init; }

    /// <summary>
    /// Gets whether the kernel has been compiled
    /// </summary>
    public bool IsCompiled { get; private set; }

    /// <summary>
    /// Gets the compiled kernel (null if not yet compiled)
    /// </summary>
    public CompiledKernel? CompiledKernel { get; private set; }

    /// <summary>
    /// Gets the compilation time in milliseconds
    /// </summary>
    public long CompilationTimeMs { get; private set; }

    /// <summary>
    /// Creates a new lazy compilation context
    /// </summary>
    public LazyCompilationContext()
    {
        IsCompiled = false;
        CompilationTimeMs = 0;
    }

    /// <summary>
    /// Ensures the kernel is compiled (compiles if not already compiled)
    /// Thread-safe to prevent duplicate compilations
    /// </summary>
    /// <param name="compiler">The kernel compiler to use</param>
    public void EnsureCompiled(IKernelCompiler compiler)
    {
        if (IsCompiled)
        {
            return;
        }

        lock (_lockObject)
        {
            // Double-check pattern
            if (IsCompiled)
            {
                return;
            }

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            CompiledKernel = compiler.Compile(Operation, InputShapes, OutputShapes);
            IsCompiled = true;

            stopwatch.Stop();
            CompilationTimeMs = stopwatch.ElapsedMilliseconds;
        }
    }

    /// <summary>
    /// Executes the kernel (ensures compilation first)
    /// </summary>
    /// <param name="executor">The kernel executor</param>
    /// <param name="inputs">Input tensors</param>
    /// <param name="outputs">Output tensors</param>
    public void Execute(IKernelExecutor executor, List<Tensor> inputs, List<Tensor> outputs)
    {
        if (CompiledKernel == null)
        {
            throw new InvalidOperationException("Kernel must be compiled before execution. Call EnsureCompiled first.");
        }

        executor.Execute(CompiledKernel, inputs, outputs);
    }

    /// <summary>
    /// Creates a new lazy compilation context
    /// </summary>
    public static LazyCompilationContext Create(Operation operation, List<int[]> inputShapes, List<int[]> outputShapes)
    {
        return new LazyCompilationContext
        {
            Operation = operation,
            InputShapes = inputShapes,
            OutputShapes = outputShapes
        };
    }
}
