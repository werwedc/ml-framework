namespace MLFramework.Fusion;

/// <summary>
/// Interface for CUDA optimization passes
/// </summary>
public interface ICudaOptimizationPass
{
    /// <summary>
    /// Applies optimization to template context
    /// </summary>
    TemplateContext Optimize(TemplateContext context, GenerationOptions options);
}
