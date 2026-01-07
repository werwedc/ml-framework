using MLFramework.Shapes;
using MLFramework.Core;

namespace MLFramework.Fusion.Dynamic;

/// <summary>
/// Generates fused kernels for fusion nodes
/// </summary>
public class FusionKernelGenerator
{
    private readonly Dictionary<string, CompiledKernel> _kernelCache = new();

    /// <summary>
    /// Generates a specialized fused kernel for the given node and concrete shapes
    /// </summary>
    /// <param name="node">The fusion node</param>
    /// <param name="concreteShapes">The concrete shapes to specialize for</param>
    /// <returns>A compiled kernel specialized for the given shapes</returns>
    public CompiledKernel GenerateFusedKernel(FusionNode node, List<int[]> concreteShapes)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        if (concreteShapes == null || concreteShapes.Count == 0)
            throw new ArgumentException("Concrete shapes cannot be null or empty", nameof(concreteShapes));

        var signature = GenerateSignature(node, concreteShapes);

        // Check cache
        if (_kernelCache.TryGetValue(signature, out var cachedKernel))
        {
            return cachedKernel;
        }

        // Generate new kernel
        var kernel = CreateSpecializedKernel(node, concreteShapes, signature);
        _kernelCache[signature] = kernel;

        return kernel;
    }

    /// <summary>
    /// Generates a generic fused kernel that works with any shape
    /// </summary>
    /// <param name="node">The fusion node</param>
    /// <returns A generic compiled kernel</returns>
    public CompiledKernel GenerateGenericKernel(FusionNode node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        var signature = $"generic_{node.GetFusedSignature()}";

        // Check cache
        if (_kernelCache.TryGetValue(signature, out var cachedKernel))
        {
            return cachedKernel;
        }

        // Generate new generic kernel
        var kernel = CreateGenericKernel(node, signature);
        _kernelCache[signature] = kernel;

        return kernel;
    }

    /// <summary>
    /// Determines whether a specialized kernel can be generated for the given shapes
    /// </summary>
    /// <param name="shapes">The shapes to check</param>
    /// <returns>True if a specialized kernel can be generated; otherwise, false</returns>
    public bool CanGenerateSpecialized(List<int[]> shapes)
    {
        if (shapes == null || shapes.Count == 0)
            return false;

        // All shapes must be non-null and have at least one dimension
        return shapes.All(shape => shape != null && shape.Length > 0);
    }

    /// <summary>
    /// Clears the kernel cache
    /// </summary>
    public void ClearCache()
    {
        _kernelCache.Clear();
    }

    /// <summary>
    /// Gets the number of cached kernels
    /// </summary>
    public int CacheCount => _kernelCache.Count;

    /// <summary>
    /// Generates a signature for a fusion node with concrete shapes
    /// </summary>
    private string GenerateSignature(FusionNode node, List<int[]> concreteShapes)
    {
        var opTypes = string.Join("->", node.Operations.Select(op => op.Type));
        var shapesSig = string.Join("|", concreteShapes.Select(s => $"[{string.Join(",", s)}]"));

        return $"specialized_{opTypes}:{shapesSig}";
    }

    /// <summary>
    /// Creates a specialized kernel for the given node and shapes
    /// </summary>
    private CompiledKernel CreateSpecializedKernel(FusionNode node, List<int[]> concreteShapes, string signature)
    {
        var sourceCode = GenerateKernelSource(node, concreteShapes, isGeneric: false);
        var binary = CompileSourceToBinary(sourceCode);

        return new CompiledKernel
        {
            KernelId = Guid.NewGuid().ToString("N"),
            SourceCode = sourceCode,
            Binary = binary,
            SpecializedShapes = concreteShapes,
            IsGeneric = false,
            Signature = signature,
            EstimatedExecutionTimeNs = EstimateExecutionTime(node, isGeneric: false)
        };
    }

    /// <summary>
    /// Creates a generic kernel for the given node
    /// </summary>
    private CompiledKernel CreateGenericKernel(FusionNode node, string signature)
    {
        var sourceCode = GenerateKernelSource(node, null, isGeneric: true);
        var binary = CompileSourceToBinary(sourceCode);

        return new CompiledKernel
        {
            KernelId = Guid.NewGuid().ToString("N"),
            SourceCode = sourceCode,
            Binary = binary,
            SpecializedShapes = new List<int[]>(),
            IsGeneric = true,
            Signature = signature,
            EstimatedExecutionTimeNs = EstimateExecutionTime(node, isGeneric: true)
        };
    }

    /// <summary>
    /// Generates kernel source code for the fusion node
    /// </summary>
    private string GenerateKernelSource(FusionNode node, List<int[]>? concreteShapes, bool isGeneric)
    {
        var builder = new System.Text.StringBuilder();

        builder.AppendLine("// Fused Kernel Source Code");
        builder.AppendLine($"// Fusion ID: {node.FusionId}");
        builder.AppendLine($"// Operations: {string.Join(" -> ", node.Operations.Select(op => op.Type))}");
        builder.AppendLine();

        if (isGeneric)
        {
            builder.AppendLine("// Generic kernel - works with any shape");
            builder.AppendLine("template<typename T>");
            builder.AppendLine("__global__ void fused_kernel_generic(T* inputs, T* outputs, const int* dims, int rank) {");
            builder.AppendLine("    // Generic implementation using runtime shape information");
            builder.AppendLine("    int idx = blockIdx.x * blockDim.x + threadIdx.x;");
            builder.AppendLine("    int total = 1;");
            builder.AppendLine("    for (int i = 0; i < rank; i++) total *= dims[i];");
            builder.AppendLine("    if (idx >= total) return;");
            builder.AppendLine("    // Fused operations implementation");
            builder.AppendLine("    outputs[idx] = inputs[idx]; // Placeholder");
            builder.AppendLine("}");
        }
        else
        {
            builder.AppendLine("// Specialized kernel for specific shapes");
            var shapeStr = string.Join(", ", concreteShapes!.Select(s => $"[{string.Join(",", s)}]"));
            builder.AppendLine($"// Shapes: {shapeStr}");

            // Generate specialized code based on operation types
            builder.AppendLine("__global__ void fused_kernel_specialized(float* inputs, float* outputs) {");
            builder.AppendLine("    int idx = blockIdx.x * blockDim.x + threadIdx.x;");

            // Generate optimized code for the specific shape
            if (concreteShapes.Count > 0 && concreteShapes[0].Length > 0)
            {
                var totalElements = concreteShapes[0].Aggregate(1, (acc, dim) => acc * dim);
                builder.AppendLine($"    if (idx >= {totalElements}) return;");
            }

            builder.AppendLine("    // Fused operations implementation");
            builder.AppendLine("    outputs[idx] = inputs[idx]; // Placeholder");
            builder.AppendLine("}");
        }

        return builder.ToString();
    }

    /// <summary>
    /// Compiles source code to binary (placeholder implementation)
    /// </summary>
    private byte[] CompileSourceToBinary(string sourceCode)
    {
        // In a real implementation, this would invoke a compiler (e.g., CUDA compiler)
        // For now, we return a placeholder binary
        return System.Text.Encoding.UTF8.GetBytes(sourceCode);
    }

    /// <summary>
    /// Estimates execution time for the kernel
    /// </summary>
    private long EstimateExecutionTime(FusionNode node, bool isGeneric)
    {
        // Simple estimation: more operations = longer time
        // Generic kernels are slower than specialized ones
        var baseTime = 1000L; // 1 microsecond base time
        var opTime = node.Operations.Count * 500L; // 0.5 microseconds per operation
        var genericMultiplier = isGeneric ? 2.0 : 1.0;

        return (long)((baseTime + opTime) * genericMultiplier);
    }
}
