using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// CUDA-specific kernel code generator for fused operations
/// </summary>
public class CudaKernelGenerator : BackendSpecificGenerator
{
    private readonly ICudaOptimizationPass _optimizer;
    private readonly ISharedMemoryPlanner _memoryPlanner;

    public CudaKernelGenerator(
        ICodeTemplateEngine templateEngine,
        ICudaOptimizationPass optimizer,
        ISharedMemoryPlanner memoryPlanner)
        : base(templateEngine)
    {
        _optimizer = optimizer;
        _memoryPlanner = memoryPlanner;
        BackendType = KernelBackendType.CUDA;
    }

    public override KernelBackendType BackendType { get; }

    public override bool CanCompile(FusedOperation fusedOp)
    {
        // Check if all operations are supported
        return fusedOp.ConstituentOperations.All(op => IsOperationSupported(op.Type));
    }

    private bool IsOperationSupported(string opType)
    {
        return opType switch
        {
            "Add" or "Sub" or "Mul" or "Div" => true,
            "ReLU" or "Sigmoid" or "Tanh" or "LeakyReLU" => true,
            "Exp" or "Log" or "Sqrt" or "Abs" => true,
            "Conv2D" or "Linear" => true,
            "BatchNorm" => true,
            "MaxPool2D" or "AvgPool2D" => true,
            _ => false
        };
    }

    protected override string GetTemplateName(FusionIR ir)
    {
        // Select template based on pattern type
        var primaryOp = ir.Nodes[0].OriginalOpType;

        return primaryOp switch
        {
            "Conv2D" when ir.Nodes.Count > 1 => "cuda_conv_activation_template",
            "Conv2D" => "cuda_conv_template",
            "Linear" => "cuda_linear_template",
            _ when IsElementWiseChain(ir) => "cuda_elementwise_template",
            _ => "cuda_generic_template"
        };
    }

    protected override string GetBackendSpecificKernelName(FusionIR ir)
    {
        var opChain = string.Join("_", ir.Nodes.Select(n => n.OriginalOpType.ToLower()));
        return $"fused_cuda_{opChain}_{ir.Id}";
    }

    protected override string GenerateSourceCode(
        FusionIR ir,
        IReadOnlyList<KernelParameter> parameters,
        GenerationOptions options)
    {
        var template = GetTemplate(ir);
        var context = BuildTemplateContext(ir, parameters, options);

        // Apply optimizations
        context = _optimizer.Optimize(context, options);

        // Plan shared memory usage
        if (ir.ComputeRequirements.RequiresSharedMemory)
        {
            context = context with
            {
                MemoryLayout = _memoryPlanner.PlanMemory(ir, options)
            };
        }

        return TemplateEngine.Render(template, context);
    }

    private bool IsElementWiseChain(FusionIR ir)
    {
        return ir.Nodes.All(n =>
            n.OriginalOpType is "Add" or "Sub" or "Mul" or "Div" or
            "ReLU" or "Sigmoid" or "Tanh" or "Exp" or "Log" or "Sqrt" or "Abs");
    }
}
