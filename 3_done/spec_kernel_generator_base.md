# Spec: Kernel Code Generator Base

## Overview
Implement the base infrastructure for kernel code generation, including abstractions for different backends, template systems, and kernel specification handling.

## Requirements

### 1. Kernel Generator Interface
Base interface for generating fused kernel code.

```csharp
public interface IKernelCodeGenerator
{
    /// <summary>
    /// Generates kernel code for a fused operation
    /// </summary>
    KernelCodeResult GenerateKernel(FusedOperation fusedOp, GenerationOptions options);

    /// <summary>
    /// Gets supported backend type
    /// </summary>
    KernelBackendType BackendType { get; }

    /// <summary>
    /// Validates that a fused operation can be compiled by this backend
    /// </summary>
    bool CanCompile(FusedOperation fusedOp);
}

public enum KernelBackendType
{
    CUDA,
    HIP,
    Triton,
    CUDAPlus,
    Metal,
    OpenCL
}

public record KernelCodeResult
{
    public required string KernelSourceCode { get; init; }
    public required string KernelName { get; init; }
    public required IReadOnlyList<KernelParameter> Parameters { get; init; }
    public required CompilationMetadata Metadata { get; init; }
    public required IReadOnlyList<string> Includes { get; init; }
}

public record KernelParameter
{
    public required string Name { get; init; }
    public required KernelParameterType Type { get; init; }
    public required ParameterDirection Direction { get; init; }
    public required TensorDataType DataType { get; init; }
}

public enum KernelParameterType
{
    Tensor,
    Scalar,
    Pointer,
    Int,
    Float
}

public enum ParameterDirection
{
    Input,
    Output,
    InputOutput
}

public record CompilationMetadata
{
    public required int SharedMemoryBytes { get; init; }
    public required int RegisterCount { get; init; }
    public required int ThreadBlockSize { get; init; }
    public required int GridSize { get; init; }
    public required IReadOnlySet<string> RequiredCapabilities { get; init; }
}

public record GenerationOptions
{
    public KernelBackendType TargetBackend { get; init; } = KernelBackendType.CUDA;
    public OptimizationLevel OptimizationLevel { get; init; } = OptimizationLevel.O3;
    public bool EnableVectorization { get; init; } = true;
    public bool EnableSharedMemory { get; init; } = true;
    public bool EnableUnrolling { get; init; } = true;
    public int ComputeCapability { get; init; } = 75; // SM_7.5
}

public enum OptimizationLevel
{
    None,
    O0,
    O1,
    O2,
    O3,
    Fast
}
```

### 2. Abstract Base Generator
Base implementation with common functionality.

```csharp
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
                Type = KernelParameterType.Tensor,
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

        if (ir.Variables.Any(v => v.DataType == TensorDataType.Float16))
            capabilities.Add("fp16");

        if (ir.Variables.Any(v => v.DataType == TensorDataType.BFloat16))
            capabilities.Add("bf16");

        if (ir.ComputeRequirements.RequiresSharedMemory)
            capabilities.Add("shared_mem");

        return capabilities;
    }
}
```

### 3. Code Template Engine
Template system for generating kernel code.

```csharp
public interface ICodeTemplateEngine
{
    string Render(string template, TemplateContext context);
    string LoadTemplate(string templateName);
    void RegisterTemplate(string templateName, string templateContent);
}

public record TemplateContext
{
    public required string KernelName { get; init; }
    public required IReadOnlyList<KernelParameter> Parameters { get; init; }
    public required IReadOnlyList<FusionOpNode> Nodes { get; init; }
    public required MemoryLayout MemoryLayout { get; init; }
    public required ComputeRequirements ComputeRequirements { get; init; }
    public required GenerationOptions Options { get; init; }
}

public class CodeTemplateEngine : ICodeTemplateEngine
{
    private readonly Dictionary<string, string> _templates = new();
    private readonly ITemplateRenderer _renderer;

    public CodeTemplateEngine(ITemplateRenderer renderer)
    {
        _renderer = renderer;
    }

    public void RegisterTemplate(string templateName, string templateContent)
    {
        _templates[templateName] = templateContent;
    }

    public string LoadTemplate(string templateName)
    {
        if (!_templates.TryGetValue(templateName, out var template))
            throw new KeyNotFoundException($"Template '{templateName}' not found");

        return template;
    }

    public string Render(string template, TemplateContext context)
    {
        return _renderer.Render(template, context);
    }
}

public interface ITemplateRenderer
{
    string Render(string template, TemplateContext context);
}

public class SimpleTemplateRenderer : ITemplateRenderer
{
    public string Render(string template, TemplateContext context)
    {
        // Simple placeholder replacement
        var result = template;

        result = result.Replace("{{kernel_name}}", context.KernelName);
        result = result.Replace("{{shared_memory}}", context.MemoryLayout.SharedMemoryBytes.ToString());
        result = result.Replace("{{threads_per_block}}", context.ComputeRequirements.ThreadsPerBlock.ToString());

        // Generate parameter list
        var paramList = string.Join(", ", context.Parameters.Select(p =>
            $"{GetCudaTypeName(p.DataType)}* {p.Name}"));
        result = result.Replace("{{parameters}}", paramList);

        // Generate kernel body
        var body = GenerateKernelBody(context);
        result = result.Replace("{{kernel_body}}", body);

        return result;
    }

    private string GetCudaTypeName(TensorDataType dtype)
    {
        return dtype switch
        {
            TensorDataType.Float32 => "float",
            TensorDataType.Float16 => "half",
            TensorDataType.BFloat16 => "__nv_bfloat16",
            TensorDataType.Int32 => "int",
            TensorDataType.Int64 => "long long",
            TensorDataType.Int16 => "short",
            TensorDataType.Int8 => "char",
            TensorDataType.UInt32 => "unsigned int",
            _ => throw new NotSupportedException($"Unsupported dtype: {dtype}")
        };
    }

    private string GenerateKernelBody(TemplateContext context)
    {
        var sb = new StringBuilder();

        foreach (var node in context.Nodes)
        {
            sb.AppendLine(GenerateNodeCode(node, context));
        }

        return sb.ToString();
    }

    private string GenerateNodeCode(FusionOpNode node, TemplateContext context)
    {
        return node.OriginalOpType switch
        {
            "Add" => GenerateAddCode(node),
            "Mul" => GenerateMulCode(node),
            "ReLU" => GenerateReLUCode(node),
            "Sigmoid" => GenerateSigmoidCode(node),
            _ => $"// Unknown op type: {node.OriginalOpType}"
        };
    }

    private string GenerateAddCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = {node.InputVars[0]} + {node.InputVars[1]};";
    }

    private string GenerateMulCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = {node.InputVars[0]} * {node.InputVars[1]};";
    }

    private string GenerateReLUCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = fmaxf(0.0f, {node.InputVars[0]});";
    }

    private string GenerateSigmoidCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = 1.0f / (1.0f + expf(-{node.InputVars[0]}));";
    }
}
```

### 4. Backend Abstraction
Support multiple backend targets.

```csharp
public abstract class BackendSpecificGenerator : KernelCodeGeneratorBase
{
    protected BackendSpecificGenerator(ICodeTemplateEngine templateEngine)
        : base(templateEngine)
    {
    }

    protected override string GetTemplate(FusionIR ir)
    {
        return LoadTemplate(GetTemplateName(ir));
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

    protected override string GetCudaTypeName(TensorDataType dtype)
    {
        // Backend-specific type mapping
        return BackendType switch
        {
            KernelBackendType.CUDA => GetCudaTypeName(dtype),
            KernelBackendType.HIP => GetHipTypeName(dtype),
            KernelBackendType.Triton => GetTritonTypeName(dtype),
            _ => base.GetCudaTypeName(dtype)
        };
    }

    protected virtual string GetHipTypeName(TensorDataType dtype)
    {
        // HIP uses same types as CUDA
        return GetCudaTypeName(dtype);
    }

    protected virtual string GetTritonTypeName(TensorDataType dtype)
    {
        return dtype switch
        {
            TensorDataType.Float32 => "tl.float32",
            TensorDataType.Float16 => "tl.float16",
            TensorDataType.BFloat16 => "tl.bfloat16",
            TensorDataType.Int32 => "tl.int32",
            _ => throw new NotSupportedException($"Unsupported dtype: {dtype}")
        };
    }
}
```

### 5. Kernel Specification
Detailed specification for kernel compilation and execution.

```csharp
public record KernelSpecification
{
    public required string KernelName { get; init; }
    public required FusionStrategy Strategy { get; init; }
    public required IReadOnlyList<FusionVariable> InputTensors { get; init; }
    public required IReadOnlyList<FusionVariable> OutputTensors { get; init; }
    public required int TemporaryMemoryBytes { get; init; }
    public required int RegisterBytes { get; init; }
    public required ThreadBlockConfiguration ThreadBlockConfig { get; init; }
    public required IReadOnlyList<string> CompilationFlags { get; init; }
}

public record ThreadBlockConfiguration
{
    public required int X { get; init; }
    public required int Y { get; init; }
    public required int Z { get; init; }
    public int Total => X * Y * Z;
}

public record KernelLaunchConfiguration
{
    public required ThreadBlockConfiguration BlockDim { get; init; }
    public required ThreadBlockConfiguration GridDim { get; init; }
    public required int SharedMemoryBytes { get; init; }
    public required IReadOnlyList<KernelLaunchParameter> Parameters { get; init; }
}

public record KernelLaunchParameter
{
    public required string Name { get; init; }
    public required object Value { get; init; }
    public required KernelParameterType Type { get; init; }
}
```

## Implementation Tasks

1. **Create kernel generator interfaces and records** (25 min)
   - IKernelCodeGenerator interface
   - KernelBackendType enum
   - KernelCodeResult and related records
   - GenerationOptions and enums

2. **Implement KernelCodeGeneratorBase** (30 min)
   - Abstract base class
   - GenerateKernel method
   - Parameter computation
   - Metadata computation

3. **Implement CodeTemplateEngine** (25 min)
   - ICodeTemplateEngine interface
   - Template registration and loading
   - TemplateContext record

4. **Implement SimpleTemplateRenderer** (30 min)
   - Placeholder replacement
   - Type name mapping
   - Kernel body generation
   - Operation-specific code generation

5. **Implement BackendSpecificGenerator** (20 min)
   - Backend abstraction
   - Backend-specific kernel naming
   - Backend-specific type mapping

6. **Define KernelSpecification and related types** (20 min)
   - KernelSpecification record
   - ThreadBlockConfiguration record
   - KernelLaunchConfiguration record

## Test Cases

```csharp
[Test]
public void GenerateKernel_ProducesValidCode()
{
    var generator = CreateMockGenerator();
    var fusedOp = CreateSimpleFusedOperation();
    var options = new GenerationOptions();

    var result = generator.GenerateKernel(fusedOp, options);

    Assert.IsNotNull(result.KernelSourceCode);
    Assert.IsNotEmpty(result.KernelSourceCode);
    Assert.IsNotEmpty(result.Parameters);
}

[Test]
public void TemplateRenderer_ReplacesPlaceholders()
{
    var renderer = new SimpleTemplateRenderer();
    var template = "{{kernel_name}} - {{shared_memory}}";
    var context = new TemplateContext
    {
        KernelName = "test_kernel",
        MemoryLayout = new MemoryLayout { SharedMemoryBytes = 1024 },
        // ... other required fields
    };

    var result = renderer.Render(template, context);

    Assert.IsTrue(result.Contains("test_kernel"));
    Assert.IsTrue(result.Contains("1024"));
}

[Test]
public void ComputeParameters_CorrectForIR()
{
    var ir = CreateFusionIRWithVariables();
    var generator = new TestKernelCodeGenerator();

    var parameters = generator.ComputeKernelParameters(ir);

    Assert.AreEqual(2, parameters.Count);
    Assert.AreEqual(ParameterDirection.Input, parameters[0].Direction);
    Assert.AreEqual(ParameterDirection.Output, parameters[1].Direction);
}

[Test]
public void CanCompile_SupportedOp_ReturnsTrue()
{
    var generator = CreateMockGenerator();
    var fusedOp = CreateFusedOpWithSupportedOps();

    Assert.IsTrue(generator.CanCompile(fusedOp));
}
```

## Success Criteria
- Kernel code generators produce syntactically correct code
- Template engine correctly renders templates
- Backend abstraction works for multiple backends
- Kernel specification is complete and accurate
- Parameters and metadata are correctly computed

## Dependencies
- FusedOperation from fusion engine
- FusionIR from fusion engine
- Template infrastructure (external or custom)
