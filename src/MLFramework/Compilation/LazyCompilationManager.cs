using MLFramework.Fusion;
using MLFramework.Fusion.Dynamic;

namespace MLFramework.Compilation;

/// <summary>
/// Manages lazy compilation of kernels
/// </summary>
public class LazyCompilationManager
{
    private readonly IKernelCache<CompiledKernel> _cache;
    private readonly IKernelCompiler _compiler;
    private readonly CompilationStats _stats;
    private readonly HashSet<string> _uniqueKernelSignatures;

    /// <summary>
    /// Gets the kernel cache
    /// </summary>
    public IKernelCache<CompiledKernel> Cache => _cache;

    /// <summary>
    /// Gets the kernel compiler
    /// </summary>
    public IKernelCompiler Compiler => _compiler;

    /// <summary>
    /// Creates a new lazy compilation manager
    /// </summary>
    /// <param name="cache">The kernel cache to use</param>
    /// <param name="compiler">The kernel compiler to use</param>
    public LazyCompilationManager(IKernelCache<CompiledKernel> cache, IKernelCompiler compiler)
    {
        _cache = cache ?? throw new ArgumentNullException(nameof(cache));
        _compiler = compiler ?? throw new ArgumentNullException(nameof(compiler));
        _stats = new CompilationStats();
        _uniqueKernelSignatures = new HashSet<string>();
    }

    /// <summary>
    /// Gets or compiles a kernel for the given operation and shapes
    /// </summary>
    /// <param name="op">The operation to compile</param>
    /// <param name="inputShapes">Input tensor shapes</param>
    /// <returns>A compiled kernel</returns>
    public CompiledKernel GetOrCompile(Operation op, List<int[]> inputShapes)
    {
        var signature = ShapeSignature.Create(op.Type, inputShapes);

        // Try to get from cache
        var cachedKernel = _cache.Get(signature);
        if (cachedKernel != null)
        {
            lock (_stats)
            {
                _stats.CacheHits++;
            }
            return cachedKernel;
        }

        // Cache miss - compile the kernel
        lock (_stats)
        {
            _stats.CacheMisses++;
        }

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        // Infer output shapes (simplified - in practice this would use shape inference)
        var outputShapes = InferOutputShapes(op, inputShapes);

        var kernel = _compiler.Compile(op, inputShapes, outputShapes);

        stopwatch.Stop();

        // Update cache and stats
        _cache.Set(signature, kernel);

        lock (_stats)
        {
            _stats.TotalCompilations++;
            _stats.TotalCompilationTimeMs += stopwatch.ElapsedMilliseconds;
            _uniqueKernelSignatures.Add(kernel.KernelId);
            _stats.UniqueKernels = _uniqueKernelSignatures.Count;
        }

        return kernel;
    }

    /// <summary>
    /// Precompiles kernels for multiple shape variants to warm up the cache
    /// </summary>
    /// <param name="op">The operation to compile</param>
    /// <param name="shapeVariants">List of shape variants to precompile</param>
    public void Precompile(Operation op, List<List<int[]>> shapeVariants)
    {
        foreach (var shapes in shapeVariants)
        {
            try
            {
                GetOrCompile(op, shapes);
            }
            catch (Exception ex)
            {
                // Log and continue - don't fail entire precompilation
                Console.WriteLine($"Warning: Failed to precompile kernel for shapes {shapes}: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Clears the kernel cache
    /// </summary>
    public void ClearCache()
    {
        _cache.Clear();
        _stats.UniqueKernels = 0;
        _uniqueKernelSignatures.Clear();
    }

    /// <summary>
    /// Gets compilation statistics
    /// </summary>
    /// <returns>Current compilation statistics</returns>
    public CompilationStats GetCompilationStats()
    {
        lock (_stats)
        {
            return new CompilationStats
            {
                TotalCompilations = _stats.TotalCompilations,
                CacheHits = _stats.CacheHits,
                CacheMisses = _stats.CacheMisses,
                TotalCompilationTimeMs = _stats.TotalCompilationTimeMs,
                UniqueKernels = _stats.UniqueKernels
            };
        }
    }

    /// <summary>
    /// Creates a lazy compilation context for the given operation and shapes
    /// </summary>
    /// <param name="op">The operation</param>
    /// <param name="inputShapes">Input shapes</param>
    /// <returns>A lazy compilation context</returns>
    public LazyCompilationContext CreateContext(Operation op, List<int[]> inputShapes)
    {
        var outputShapes = InferOutputShapes(op, inputShapes);
        return LazyCompilationContext.Create(op, inputShapes, outputShapes);
    }

    /// <summary>
    /// Infers output shapes for the given operation and input shapes
    /// </summary>
    private List<int[]> InferOutputShapes(Operation op, List<int[]> inputShapes)
    {
        // Simplified shape inference - in practice this would use a proper shape inference engine
        // For now, assume output shapes match input shapes (identity operation)
        return inputShapes.Select(s => s.ToArray()).ToList();
    }
}
