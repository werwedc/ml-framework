using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Abstract class for backend-specific kernel code generation
/// </summary>
public abstract class BackendSpecificGenerator : KernelCodeGeneratorBase
{
    protected BackendSpecificGenerator(ICodeTemplateEngine templateEngine)
        : base(templateEngine)
    {
    }

    protected override string GetTemplate(FusionIR ir)
    {
        return TemplateEngine.LoadTemplate(GetTemplateName(ir));
    }

    protected abstract string GetTemplateName(FusionIR ir);

    protected override TemplateContext BuildTemplateContext(
        FusionIR ir,
        IReadOnlyList<KernelParameter> parameters,
        GenerationOptions options)
    {
        return new TemplateContext
        {
            KernelName = GetBackendSpecificKernelName(ir),
            Parameters = parameters,
            Nodes = ir.Nodes,
            MemoryLayout = ir.MemoryLayout,
            ComputeRequirements = ir.ComputeRequirements,
            Options = options
        };
    }

    protected abstract string GetBackendSpecificKernelName(FusionIR ir);

    protected string GetCudaTypeName(DataType dtype)
    {
        // Backend-specific type mapping
        return BackendType switch
        {
            KernelBackendType.CUDA => GetCudaTypeNameImpl(dtype),
            KernelBackendType.HIP => GetHipTypeName(dtype),
            KernelBackendType.Triton => GetTritonTypeName(dtype),
            _ => GetCudaTypeNameImpl(dtype)
        };
    }

    protected virtual string GetCudaTypeNameImpl(DataType dtype)
    {
        return dtype switch
        {
            DataType.Float32 => "float",
            DataType.Float16 => "half",
            DataType.BFloat16 => "__nv_bfloat16",
            DataType.Int32 => "int",
            DataType.Int64 => "long long",
            DataType.Int16 => "short",
            DataType.Int8 => "sbyte",
            DataType.UInt8 => "unsigned char",
            DataType.Bool => "bool",
            _ => throw new NotSupportedException($"Unsupported dtype: {dtype}")
        };
    }

    protected virtual string GetHipTypeName(DataType dtype)
    {
        // HIP uses same types as CUDA
        return GetCudaTypeNameImpl(dtype);
    }

    protected virtual string GetTritonTypeName(DataType dtype)
    {
        return dtype switch
        {
            DataType.Float32 => "tl.float32",
            DataType.Float16 => "tl.float16",
            DataType.BFloat16 => "tl.bfloat16",
            DataType.Int32 => "tl.int32",
            _ => throw new NotSupportedException($"Unsupported dtype: {dtype}")
        };
    }
}
