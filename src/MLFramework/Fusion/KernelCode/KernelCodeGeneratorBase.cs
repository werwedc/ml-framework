using System.Text;
using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Abstract base implementation with common functionality for kernel code generation
/// </summary>
public abstract class KernelCodeGeneratorBase : IKernelCodeGenerator
{
    protected readonly ICodeTemplateEngine TemplateEngine;

    protected KernelCodeGeneratorBase(ICodeTemplateEngine templateEngine)
    {
        TemplateEngine = templateEngine;
    }

    public abstract KernelBackendType BackendType { get; }

    public abstract bool CanCompile(FusedOperation fusedOp);

    public virtual KernelCodeResult GenerateKernel(FusedOperation fusedOp, GenerationOptions options)
    {
        // Validate inputs
        if (!CanCompile(fusedOp))
            throw new InvalidOperationException("Cannot compile this fused operation");

        // Extract IR
        var ir = fusedOp.IntermediateRepresentation;

        // Compute kernel parameters
        var parameters = ComputeKernelParameters(ir);

        // Compute compilation metadata
        var metadata = ComputeCompilationMetadata(ir, options);

        // Generate code
        var sourceCode = GenerateSourceCode(ir, parameters, options);

        // Compute includes
        var includes = ComputeIncludes(ir, options);

        return new KernelCodeResult
        {
            KernelSourceCode = sourceCode,
            KernelName = $"fused_{ir.Id}",
            Parameters = parameters,
            Metadata = metadata,
            Includes = includes
        };
    }

    protected virtual string GenerateSourceCode(
        FusionIR ir,
        IReadOnlyList<KernelParameter> parameters,
        GenerationOptions options)
    {
        var template = GetTemplate(ir);
        var context = BuildTemplateContext(ir, parameters, options);
        return TemplateEngine.Render(template, context);
    }

    protected abstract string GetTemplate(FusionIR ir);

    protected abstract TemplateContext BuildTemplateContext(
        FusionIR ir,
        IReadOnlyList<KernelParameter> parameters,
        GenerationOptions options);

    protected virtual IReadOnlyList<KernelParameter> ComputeKernelParameters(FusionIR ir)
    {
        var parameters = new List<KernelParameter>();

        foreach (var variable in ir.Variables)
        {
            parameters.Add(new KernelParameter
            {
                Name = variable.Name,
                Direction = variable.Location switch
                {
                    MemoryLocation.Input => ParameterDirection.Input,
                    MemoryLocation.Output => ParameterDirection.Output,
                    _ => ParameterDirection.InputOutput
                },
                DataType = variable.DataType
            });
        }

        return parameters;
    }

    protected virtual CompilationMetadata ComputeCompilationMetadata(
        FusionIR ir,
        GenerationOptions options)
    {
        var requirements = ir.ComputeRequirements;

        return new CompilationMetadata
        {
            SharedMemoryBytes = ir.MemoryLayout.SharedMemoryBytes,
            RegisterCount = ir.MemoryLayout.RegisterBytes / 4, // Assuming 4-byte registers
            ThreadBlockSize = requirements.ThreadsPerBlock,
            GridSize = requirements.ThreadBlocks,
            RequiredCapabilities = ComputeRequiredCapabilities(ir)
        };
    }

    protected virtual IReadOnlyList<string> ComputeIncludes(FusionIR ir, GenerationOptions options)
    {
        var includes = new List<string>
        {
            "<cuda_fp16.h>",
            "<cuda_fp16.hpp>",
            "<cuda_bf16.h>"
        };

        return includes;
    }

    protected virtual IReadOnlySet<string> ComputeRequiredCapabilities(FusionIR ir)
    {
        var capabilities = new HashSet<string>();

        if (ir.Variables.Any(v => v.DataType == DataType.Float16))
            capabilities.Add("fp16");

        if (ir.Variables.Any(v => v.DataType == DataType.BFloat16))
            capabilities.Add("bf16");

        if (ir.ComputeRequirements.RequiresSharedMemory)
            capabilities.Add("shared_mem");

        return capabilities;
    }
}
