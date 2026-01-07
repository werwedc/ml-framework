# Spec: Backend Abstraction Layer

## Overview
Define the backend abstraction that allows the IR system to target different hardware platforms (CPU, CUDA, ROCm, Metal). This provides a common interface for all backends while allowing backend-specific optimizations.

## Requirements

### Backend Interface

```csharp
public interface IBackend
{
    string Name { get; }
    string TargetTriple { get; }

    // Compilation
    bool CanCompile(HLIRModule module);
    void Compile(HLIRModule module, CompilationOptions options);

    // Lowering
    HLIRModule LowerToBackendIR(HLIRModule module);

    // Code generation
    string GenerateCode(HLIRModule module);
    byte[] GenerateBinary(HLIRModule module);

    // Optimization
    void Optimize(HLIRModule module, OptimizationLevel level);

    // Execution (optional, for testing)
    void Execute(HLIRModule module, Dictionary<string, object> inputs);
}

public enum OptimizationLevel
{
    None,
    Basic,
    Standard,
    Aggressive
}
```

### Compilation Options

```csharp
public class CompilationOptions
{
    public OptimizationLevel OptimizationLevel { get; set; } = OptimizationLevel.Standard;
    public bool DebugSymbols { get; set; } = false;
    public bool Verbose { get; set; } = false;
    public int VectorWidth { get; set; } = -1;  // -1 means auto-detect
    public MemoryLayout PreferredMemoryLayout { get; set; } = MemoryLayout.RowMajor;
}
```

### Backend Registry

```csharp
public class BackendRegistry
{
    private static BackendRegistry _instance;
    private Dictionary<string, IBackend> _backends;

    private BackendRegistry() { }

    public static BackendRegistry Instance
    {
        get
        {
            if (_instance == null)
                _instance = new BackendRegistry();
            return _instance;
        }
    }

    public void RegisterBackend(IBackend backend)
    {
        _backends[backend.Name] = backend;
    }

    public IBackend GetBackend(string name)
    {
        return _backends.TryGetValue(name, out var backend) ? backend : null;
    }

    public IEnumerable<IBackend> GetAllBackends() => _backends.Values;
    public bool HasBackend(string name) => _backends.ContainsKey(name);
}
```

### CPU Backend Base

```csharp
public abstract class CPUBackendBase : IBackend
{
    public abstract string Name { get; }
    public abstract string TargetTriple { get; }

    public virtual bool CanCompile(HLIRModule module) => true;
    public virtual HLIRModule LowerToBackendIR(HLIRModule module) => module;

    public virtual void Compile(HLIRModule module, CompilationOptions options)
    {
        // 1. Lower to LLIR
        // 2. Optimize
        // 3. Generate CPU-specific code
    }

    public virtual void Optimize(HLIRModule module, OptimizationLevel level)
    {
        var passManager = new IRPassManager();

        switch (level)
        {
            case OptimizationLevel.None:
                break;
            case OptimizationLevel.Basic:
                AddBasicOptimizations(passManager);
                break;
            case OptimizationLevel.Standard:
                AddStandardOptimizations(passManager);
                break;
            case OptimizationLevel.Aggressive:
                AddAggressiveOptimizations(passManager);
                break;
        }

        passManager.RunAll(module);
    }

    protected virtual void AddBasicOptimizations(IRPassManager passManager)
    {
        passManager.AddPass(new ConstantFoldingPass(), IRPassManager.PassType.Optimization);
        passManager.AddPass(new OperationSimplificationPass(), IRPassManager.PassType.Optimization);
    }

    protected virtual void AddStandardOptimizations(IRPassManager passManager)
    {
        AddBasicOptimizations(passManager);
        passManager.AddPass(new CSEPass(), IRPassManager.PassType.Optimization);
        passManager.AddPass(new DeadCodeEliminationPass(), IRPassManager.PassType.Optimization);
    }

    protected virtual void AddAggressiveOptimizations(IRPassManager passManager)
    {
        AddStandardOptimizations(passManager);
        // Add more aggressive passes (loop unrolling, vectorization, etc.)
    }

    public abstract string GenerateCode(HLIRModule module);
    public abstract byte[] GenerateBinary(HLIRModule module);
}
```

### x86 CPU Backend

```csharp
public class X86CPUBackend : CPUBackendBase
{
    public override string Name => "x86_64";
    public override string TargetTriple => "x86_64-unknown-linux-gnu";

    private CPUArchitecture _architecture;
    private bool _hasAVX2;
    private bool _hasAVX512;

    public X86CPUBackend(CPUArchitecture architecture = CPUArchitecture.X64)
    {
        _architecture = architecture;
        DetectCPUFeatures();
    }

    private void DetectCPUFeatures()
    {
        // Detect CPU features (AVX2, AVX512, etc.)
        _hasAVX2 = true;  // Placeholder
        _hasAVX512 = false;  // Placeholder
    }

    public override void Optimize(HLIRModule module, OptimizationLevel level)
    {
        base.Optimize(module, level);

        // Add x86-specific optimizations
        if (_hasAVX2)
        {
            // Add AVX2 vectorization passes
        }
    }

    public override string GenerateCode(HLIRModule module)
    {
        // Generate C code or assembly
        // For now, generate a simplified C-like representation
        var codegen = new X86CodeGenerator(this);
        return codegen.Generate(module);
    }

    public override byte[] GenerateBinary(HLIRModule module)
    {
        // Call external compiler (e.g., Clang) to generate binary
        // For now, return null or throw NotImplementedException
        throw new NotImplementedException("Binary generation not implemented");
    }
}

public enum CPUArchitecture
{
    X32,
    X64
}
```

### GPU Backend Base

```csharp
public abstract class GPUBackendBase : IBackend
{
    public abstract string Name { get; }
    public abstract string TargetTriple { get; }
    public abstract ComputeCapability ComputeCapability { get; }

    public virtual bool CanCompile(HLIRModule module) => true;

    public virtual void Compile(HLIRModule module, CompilationOptions options)
    {
        // 1. Lower to LLIR
        // 2. Optimize for GPU
        // 3. Generate GPU-specific code (e.g., PTX for CUDA)
    }

    public virtual HLIRModule LowerToBackendIR(HLIRModule module)
    {
        // Lower MLIR to GPU-specific IR
        // Insert memory transfer operations
        // Mark kernels for GPU execution
        return module;
    }

    public virtual void Optimize(HLIRModule module, OptimizationLevel level)
    {
        // GPU-specific optimizations:
        // - Kernel fusion
        // - Memory coalescing
        // - Shared memory usage
        // - Thread block size optimization
    }

    public abstract string GenerateCode(HLIRModule module);
    public abstract byte[] GenerateBinary(HLIRModule module);
}
```

### CUDA Backend

```csharp
public class CUDABackend : GPUBackendBase
{
    public override string Name => "CUDA";
    public override string TargetTriple => "nvptx64-nvidia-cuda";
    public override ComputeCapability ComputeCapability { get; }

    public CUDABackend(ComputeCapability computeCapability = ComputeCapability.SM_80)
    {
        ComputeCapability = computeCapability;
    }

    public override HLIRModule LowerToBackendIR(HLIRModule module)
    {
        var lowering = new CUDALoweringPass();
        lowering.Run(module);
        return module;
    }

    public override void Optimize(HLIRModule module, OptimizationLevel level)
    {
        base.Optimize(module, level);

        // CUDA-specific optimizations
        var cudaOptimizer = new CUDAOptimizer(this);
        cudaOptimizer.Optimize(module, level);
    }

    public override string GenerateCode(HLIRModule module)
    {
        // Generate PTX code
        var codegen = new PTXCodeGenerator(this);
        return codegen.Generate(module);
    }

    public override byte[] GenerateBinary(HLIRModule module)
    {
        // Generate cubin file (requires nvcc or ptxas)
        // For now, return null or throw NotImplementedException
        throw new NotImplementedException("Cubin generation not implemented");
    }
}

public enum ComputeCapability
{
    SM_70,  // Volta
    SM_75,  // Turing
    SM_80,  // Ampere
    SM_90   // Hopper
}
```

### Backend Compiler

```csharp
public class BackendCompiler
{
    public static CompilationResult Compile(HLIRModule module, string backendName,
                                           CompilationOptions options = null)
    {
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

public class CompilationResult
{
    public bool Success { get; set; } = true;
    public string Code { get; set; }
    public byte[] Binary { get; set; }
    public string ErrorMessage { get; set; }
}
```

## Implementation Details

1. **Backend Selection**: Backends should be selected at compile time
2. **Feature Detection**: Backends should detect available hardware features
3. **Optimization Levels**: Clear distinction between optimization levels
4. **Error Handling**: Provide clear error messages for compilation failures

## Deliverables

- `src/IR/Backend/IBackend.cs`
- `src/IR/Backend/CompilationOptions.cs`
- `src/IR/Backend/BackendRegistry.cs`
- `src/IR/Backend/CPU/CPUBackendBase.cs`
- `src/IR/Backend/CPU/X86CPUBackend.cs`
- `src/IR/Backend/CPU/CPUArchitecture.cs`
- `src/IR/Backend/GPU/GPUBackendBase.cs`
- `src/IR/Backend/GPU/CUDABackend.cs`
- `src/IR/Backend/GPU/ComputeCapability.cs`
- `src/IR/Backend/BackendCompiler.cs`
- `src/IR/Backend/CompilationResult.cs`

## Success Criteria

- Can register and retrieve backends
- CPU backend can compile a simple HLIR module
- CUDA backend infrastructure is in place
- Backend compiler produces code output

## Dependencies

- spec_ir_type_system.md
- spec_hlir_graph_builder.md
- spec_ir_transformation_infra.md
- spec_llir_foundation.md
