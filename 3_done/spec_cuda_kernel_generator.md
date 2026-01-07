# Spec: CUDA Kernel Generator

## Overview
Implement CUDA-specific kernel code generator that produces optimized CUDA/HIP kernels from fused operation IR, including memory access pattern optimization and shared memory tiling strategies.

## Requirements

### 1. CUDA Kernel Generator
Main generator class for CUDA kernels.

```csharp
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
            "Conv2D" => "cuda_conv_template",
            "Linear" => "cuda_linear_template",
            "Conv2D" when ir.Nodes.Count > 1 => "cuda_conv_activation_template",
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
}
```

### 2. CUDA Code Templates
Templates for different kernel patterns.

**Element-wise Kernel Template:**
```cuda
template<typename T>
__global__ void {{kernel_name}}(
    {{#each parameters}}
    {{> param_decl}}
    {{#unless @last}}, {{/unless}}
    {{/each}}
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements)
        return;

    // Load input
    T val = {{input_var}}[idx];

    {{#each nodes}}
    // {{original_op_type}}
    {{> op_codegen}}
    {{/each}}

    // Store output
    {{output_var}}[idx] = val;
}
```

**Conv + Activation Kernel Template:**
```cuda
template<typename T>
__global__ void {{kernel_name}}(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding
) {
    extern __shared__ T shared_mem[];

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = out_idx % out_channels;
    int out_w = (out_idx / out_channels) % out_width;
    int out_h = (out_idx / out_channels / out_width) % out_height;
    int batch_idx = out_idx / (out_channels * out_height * out_width);

    if (batch_idx >= batch_size || out_c >= out_channels ||
        out_h >= out_height || out_w >= out_width)
        return;

    T sum = bias[out_c];

    // Convolution computation
    #pragma unroll
    for (int ic = 0; ic < in_channels; ic++) {
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            int in_h = out_h * stride + kh - padding;
            if (in_h < 0 || in_h >= in_height) continue;

            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_w = out_w * stride + kw - padding;
                if (in_w < 0 || in_w >= in_width) continue;

                int input_idx = ((batch_idx * in_channels + ic) * in_height + in_h) * in_width + in_w;
                int weight_idx = ((out_c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    {{#if has_activation}}
    // Activation function
    {{activation_name}}(sum);
    {{/if}}

    output[out_idx] = sum;
}
```

### 3. Memory Access Pattern Optimization
Optimize memory access patterns for better cache utilization.

```csharp
public interface ICudaOptimizationPass
{
    TemplateContext Optimize(TemplateContext context, GenerationOptions options);
}

public class MemoryAccessOptimizer : ICudaOptimizationPass
{
    public TemplateContext Optimize(TemplateContext context, GenerationOptions options)
    {
        if (!options.EnableVectorization)
            return context;

        // Determine optimal vector width based on dtype and alignment
        var vectorWidth = ComputeOptimalVectorWidth(context, options);

        // Rewrite operations to use vectorized loads/stores
        return ApplyVectorization(context, vectorWidth);
    }

    private int ComputeOptimalVectorWidth(TemplateContext context, GenerationOptions options)
    {
        var dtype = context.Parameters[0].DataType;

        // Vector width based on data type and alignment
        return dtype switch
        {
            TensorDataType.Float32 => 4,  // float4
            TensorDataType.Float16 => 8,  // half8
            TensorDataType.Int32 => 4,    // int4
            TensorDataType.Int64 => 2,    // longlong2
            _ => 1
        };
    }

    private TemplateContext ApplyVectorization(TemplateContext context, int vectorWidth)
    {
        // Replace scalar operations with vectorized operations
        // This is a simplified version - actual implementation would need
        // more sophisticated IR transformation

        if (vectorWidth == 1)
            return context;

        // Update context with vectorized operations
        return context with
        {
            // Vectorized parameters would be generated here
            Options = context.Options with
            {
                EnableVectorization = true
            }
        };
    }
}
```

### 4. Shared Memory Tiling Strategies
Plan shared memory usage for tiling optimizations.

```csharp
public interface ISharedMemoryPlanner
{
    MemoryLayout PlanMemory(FusionIR ir, GenerationOptions options);
}

public class SharedMemoryPlanner : ISharedMemoryPlanner
{
    private const int MaxSharedMemory = 48 * 1024; // 48KB per SM

    public MemoryLayout PlanMemory(FusionIR ir, GenerationOptions options)
    {
        if (!options.EnableSharedMemory)
            return ir.MemoryLayout;

        // Calculate required shared memory for tiling
        var tilingConfig = ComputeTilingConfiguration(ir, options);

        if (tilingConfig == null)
            return ir.MemoryLayout;

        return ir.MemoryLayout with
        {
            SharedMemoryBytes = tilingConfig.TotalBytes,
            TensorLayout = OptimizeLayoutForTiling(ir, tilingConfig)
        };
    }

    private TilingConfiguration? ComputeTilingConfiguration(FusionIR ir, GenerationOptions options)
    {
        var primaryOp = ir.Nodes[0].OriginalOpType;

        return primaryOp switch
        {
            "Conv2D" => ComputeConvTiling(ir, options),
            "Linear" => ComputeLinearTiling(ir, options),
            _ when IsElementWiseChain(ir) => ComputeElementWiseTiling(ir, options),
            _ => null
        };
    }

    private TilingConfiguration? ComputeConvTiling(FusionIR ir, GenerationOptions options)
    {
        // Tiling for convolution: tile input and weights into shared memory
        var inputShape = ir.Variables[0].Shape;
        var dtypeSize = GetDataTypeSize(ir.Variables[0].DataType);

        // Tile size (configurable based on kernel size)
        int tileH = 8;
        int tileW = 8;
        int tileC = 32;

        var inputTileSize = tileH * tileW * tileC * dtypeSize;
        var weightTileSize = tileH * tileW * tileC * dtypeSize;
        var totalBytes = inputTileSize + weightTileSize;

        if (totalBytes > MaxSharedMemory)
            return null;

        return new TilingConfiguration
        {
            TileHeight = tileH,
            TileWidth = tileW,
            TileChannels = tileC,
            TotalBytes = totalBytes
        };
    }

    private TilingConfiguration? ComputeLinearTiling(FusionIR ir, GenerationOptions options)
    {
        // Tiling for matrix multiplication
        var inputShape = ir.Variables[0].Shape;
        var dtypeSize = GetDataTypeSize(ir.Variables[0].DataType);

        // Standard matrix multiplication tiling
        int tileM = 64;
        int tileN = 64;
        int tileK = 8;

        var tileSize = (tileM * tileK + tileK * tileN) * dtypeSize;
        var totalBytes = tileSize;

        if (totalBytes > MaxSharedMemory)
            return null;

        return new TilingConfiguration
        {
            TileHeight = tileM,
            TileWidth = tileN,
            TileChannels = tileK,
            TotalBytes = totalBytes
        };
    }

    private TilingConfiguration? ComputeElementWiseTiling(FusionIR ir, GenerationOptions options)
    {
        // Element-wise operations don't need shared memory tiling
        return null;
    }

    private TensorLayout OptimizeLayoutForTiling(FusionIR ir, TilingConfiguration config)
    {
        // Use NCHW for better memory coalescing
        return TensorLayout.NCHW;
    }

    private int GetDataTypeSize(TensorDataType dtype)
    {
        return dtype switch
        {
            TensorDataType.Float32 or TensorDataType.Int32 => 4,
            TensorDataType.Float16 or TensorDataType.Int16 => 2,
            TensorDataType.Int8 => 1,
            TensorDataType.Int64 => 8,
            _ => 4
        };
    }

    private bool IsElementWiseChain(FusionIR ir)
    {
        return ir.Nodes.All(n =>
            n.OriginalOpType is "Add" or "Sub" or "Mul" or "Div" or
            "ReLU" or "Sigmoid" or "Tanh" or "Exp" or "Log" or "Sqrt" or "Abs");
    }
}

public record TilingConfiguration
{
    public required int TileHeight { get; init; }
    public required int TileWidth { get; init; }
    public required int TileChannels { get; init; }
    public required int TotalBytes { get; init; }
}
```

### 5. Loop Unrolling Optimization
Apply loop unrolling for better performance.

```csharp
public class LoopUnrollingOptimizer : ICudaOptimizationPass
{
    public TemplateContext Optimize(TemplateContext context, GenerationOptions options)
    {
        if (!options.EnableUnrolling)
            return context;

        // Apply unrolling to convolution and linear operations
        return ApplyUnrolling(context, options);
    }

    private TemplateContext ApplyUnrolling(TemplateContext context, GenerationOptions options)
    {
        // Modify context to include unroll pragmas
        // This is simplified - actual implementation would analyze loops
        // and insert #pragma unroll directives

        return context with
        {
            Options = context.Options with
            {
                EnableUnrolling = true
            }
        };
    }
}
```

### 6. CUDA Type Mapping
Map tensor types to CUDA types.

```csharp
public static class CudaTypeMapper
{
    public static string GetCudaTypeName(TensorDataType dtype, int vectorWidth = 1)
    {
        var baseType = dtype switch
        {
            TensorDataType.Float32 => "float",
            TensorDataType.Float16 => "half",
            TensorDataType.BFloat16 => "__nv_bfloat16",
            TensorDataType.Int32 => "int",
            TensorDataType.Int64 => "long long",
            TensorDataType.Int16 => "short",
            TensorDataType.Int8 => "char",
            TensorDataType.UInt32 => "unsigned int",
            TensorDataType.UInt64 => "unsigned long long",
            TensorDataType.UInt16 => "unsigned short",
            TensorDataType.UInt8 => "unsigned char",
            _ => throw new NotSupportedException($"Unsupported dtype: {dtype}")
        };

        if (vectorWidth > 1)
        {
            return GetVectorTypeName(baseType, vectorWidth);
        }

        return baseType;
    }

    private static string GetVectorTypeName(string baseType, int width)
    {
        return width switch
        {
            2 => $"{baseType}2",
            3 => $"{baseType}3",
            4 => $"{baseType}4",
            8 => baseType switch
            {
                "float" => "float4",
                "half" => "half4",
                _ => $"{baseType}8"
            },
            _ => baseType
        };
    }
}
```

### 7. Operation-Specific Code Generation
Generate CUDA code for specific operations.

```csharp
public static class CudaOperationCodeGenerator
{
    public static string GenerateAddCode(FusionOpNode node)
    {
        if (node.InputVars.Count == 2)
        {
            return $"{node.OutputVar} = {node.InputVars[0]} + {node.InputVars[1]};";
        }
        else if (node.InputVars.Count == 1 && node.Attributes.ContainsKey("scalar"))
        {
            var scalar = node.Attributes["scalar"];
            return $"{node.OutputVar} = {node.InputVars[0]} + {scalar};";
        }
        else
        {
            throw new InvalidOperationException("Invalid Add operation");
        }
    }

    public static string GenerateMulCode(FusionOpNode node)
    {
        if (node.InputVars.Count == 2)
        {
            return $"{node.OutputVar} = {node.InputVars[0]} * {node.InputVars[1]};";
        }
        else if (node.InputVars.Count == 1 && node.Attributes.ContainsKey("scalar"))
        {
            var scalar = node.Attributes["scalar"];
            return $"{node.OutputVar} = {node.InputVars[0]} * {scalar};";
        }
        else
        {
            throw new InvalidOperationException("Invalid Mul operation");
        }
    }

    public static string GenerateReLUCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = fmaxf(0.0f, {node.InputVars[0]});";
    }

    public static string GenerateSigmoidCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = 1.0f / (1.0f + expf(-{node.InputVars[0]}));";
    }

    public static string GenerateTanhCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = tanhf({node.InputVars[0]});";
    }

    public static string GenerateLeakyReLUCode(FusionOpNode node)
    {
        var alpha = node.Attributes.TryGetValue("alpha", out var a) ? a : 0.01f;
        return $"float leaky_{node.Id} = {alpha}f * {node.InputVars[0]}; " +
               $"{node.OutputVar} = {node.InputVars[0]} * ({node.InputVars[0]} > 0.0f ? 1.0f : 0.0f) + " +
               $"leaky_{node.Id} * ({node.InputVars[0]} <= 0.0f ? 1.0f : 0.0f);";
    }

    public static string GenerateExpCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = expf({node.InputVars[0]});";
    }

    public static string GenerateLogCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = logf({node.InputVars[0]});";
    }

    public static string GenerateSqrtCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = sqrtf({node.InputVars[0]});";
    }

    public static string GenerateAbsCode(FusionOpNode node)
    {
        return $"{node.OutputVar} = fabsf({node.InputVars[0]});";
    }
}
```

## Implementation Tasks

1. **Implement CudaKernelGenerator** (30 min)
   - Main generator class
   - Backend-specific logic
   - Template selection based on pattern type

2. **Create CUDA kernel templates** (40 min)
   - Element-wise template
   - Conv + activation template
   - Linear template
   - Generic template

3. **Implement MemoryAccessOptimizer** (25 min)
   - Vector width computation
   - Vectorized operation generation
   - Optimization pass integration

4. **Implement SharedMemoryPlanner** (35 min)
   - Tiling configuration for conv
   - Tiling configuration for linear
   - Shared memory size management
   - Layout optimization

5. **Implement LoopUnrollingOptimizer** (15 min)
   - Unrolling optimization pass
   - Pragma generation

6. **Implement CudaTypeMapper** (10 min)
   - Type mapping for all dtypes
   - Vector type support

7. **Implement CudaOperationCodeGenerator** (25 min)
   - GenerateAddCode
   - GenerateMulCode
   - Generate activation functions
   - Generate math functions

## Test Cases

```csharp
[Test]
public void GenerateKernel_ElementWise_ProducesValidCode()
{
    var generator = CreateCudaGenerator();
    var fusedOp = CreateElementWiseFusedOperation();
    var options = new GenerationOptions { EnableVectorization = true };

    var result = generator.GenerateKernel(fusedOp, options);

    Assert.IsNotNull(result.KernelSourceCode);
    Assert.IsTrue(result.KernelSourceCode.Contains("__global__"));
    Assert.IsTrue(result.KernelSourceCode.Contains("float*"));
}

[Test]
public void MemoryAccessOptimizer_VectorizesFloat()
{
    var optimizer = new MemoryAccessOptimizer();
    var context = CreateContextWithFloat32();
    var options = new GenerationOptions { EnableVectorization = true };

    var optimized = optimizer.Optimize(context, options);

    Assert.AreEqual(4, ComputeVectorWidth(optimized)); // float4
}

[Test]
public void SharedMemoryPlanner_ComputesTilingForConv()
{
    var planner = new SharedMemoryPlanner();
    var ir = CreateConvIR();
    var options = new GenerationOptions { EnableSharedMemory = true };

    var layout = planner.PlanMemory(ir, options);

    Assert.Greater(layout.SharedMemoryBytes, 0);
    Assert.Less(layout.SharedMemoryBytes, 48 * 1024);
}

[Test]
public void GenerateReLUCode_ProducesCorrectCode()
{
    var node = CreateFusionOpNode("ReLU", new[] { "input" }, "output");
    var code = CudaOperationCodeGenerator.GenerateReLUCode(node);

    Assert.IsTrue(code.Contains("fmaxf"));
    Assert.IsTrue(code.Contains("0.0f"));
}
```

## Success Criteria
- CUDA generator produces syntactically correct CUDA code
- Memory access patterns are optimized with vectorization
- Shared memory tiling strategies are applied appropriately
- Loop unrolling is applied where beneficial
- All supported operations generate correct CUDA code
- Generated code compiles without errors

## Dependencies
- KernelCodeGeneratorBase
- FusionIR and related types
- Template engine infrastructure
- Optimization passes framework
