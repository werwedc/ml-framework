# Spec: Backend Interface

## Overview
Implement the pluggable backend interface for the fusion system, supporting multiple fusion backends (Triton, XLA, custom) with a common abstraction.

## Requirements

### 1. Backend Interface
Core interface for fusion backends.

```csharp
public interface IFusionBackend
{
    /// <summary>
    /// Gets the backend name
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the backend type
    /// </summary>
    FusionBackendType Type { get; }

    /// <summary>
    /// Checks if this backend can fuse the given operations
    /// </summary>
    bool CanFuse(IReadOnlyList<Operation> operations);

    /// <summary>
    /// Fuses the given operations
    /// </summary>
    FusionResult Fuse(IReadOnlyList<Operation> operations, FusionOptions options);

    /// <summary>
    /// Compiles a fused operation into executable kernel
    /// </summary>
    CompiledKernel Compile(FusedOperation fusedOp, CompilationOptions options);

    /// <summary>
    /// Gets backend-specific capabilities
    /// </summary>
    BackendCapabilities GetCapabilities();

    /// <summary>
    /// Initializes the backend
    /// </summary>
    void Initialize(BackendConfiguration config);
}

public enum FusionBackendType
{
    Triton,
    XLA,
    Custom,
    None
}

public record CompiledKernel
{
    public required string KernelId { get; init; }
    public required FusedOperation Operation { get; init; }
    public required KernelBinary Binary { get; init; }
    public required KernelLaunchConfiguration LaunchConfig { get; init; }
    public required CompilationMetrics Metrics { get; init; }
}

public record KernelBinary
{
    public required byte[] Data { get; init; }
    public required string Format { get; init; }
    public required IReadOnlyDictionary<string, object> Metadata { get; init; }
}

public record CompilationMetrics
{
    public required double CompilationTimeMs { get; init; }
    public required long BinarySizeBytes { get; init; }
    public required int OptimizationLevel { get; init; }
}

public record BackendConfiguration
{
    public required string DeviceId { get; init; }
    public required Dictionary<string, object> Options { get; init; }
}

public record BackendCapabilities
{
    public required IReadOnlySet<FusionPatternType> SupportedPatterns { get; init; }
    public required IReadOnlySet<TensorDataType> SupportedDataTypes { get; init; }
    public required bool SupportsAutotuning { get; init; }
    public required bool SupportsProfiling { get; init; }
    public required bool SupportsJITCompilation { get; init; }
    public required bool SupportsBinaryCache { get; init; }
}
```

### 2. Backend Registry
Registry for managing available backends.

```csharp
public interface IFusionBackendRegistry
{
    /// <summary>
    /// Registers a backend
    /// </summary>
    void Register(IFusionBackend backend);

    /// <summary>
    /// Unregisters a backend
    /// </summary>
    void Unregister(string backendName);

    /// <summary>
    /// Gets a backend by name
    /// </summary>
    IFusionBackend? GetBackend(string backendName);

    /// <summary>
    /// Gets the default backend
    /// </summary>
    IFusionBackend GetDefaultBackend();

    /// <summary>
    /// Sets the default backend
    /// </summary>
    void SetDefaultBackend(string backendName);

    /// <summary>
    /// Gets all registered backends
    /// </summary>
    IReadOnlyList<IFusionBackend> GetAllBackends();

    /// <summary>
    /// Finds a backend capable of fusing the operations
    /// </summary>
    IFusionBackend? FindCapableBackend(IReadOnlyList<Operation> operations);
}

public class FusionBackendRegistry : IFusionBackendRegistry
{
    private readonly Dictionary<string, IFusionBackend> _backends = new();
    private string _defaultBackendName = "Triton";
    private readonly object _lock = new();

    public void Register(IFusionBackend backend)
    {
        lock (_lock)
        {
            _backends[backend.Name] = backend;
        }
    }

    public void Unregister(string backendName)
    {
        lock (_lock)
        {
            _backends.Remove(backendName);
        }
    }

    public IFusionBackend? GetBackend(string backendName)
    {
        lock (_lock)
        {
            return _backends.TryGetValue(backendName, out var backend) ? backend : null;
        }
    }

    public IFusionBackend GetDefaultBackend()
    {
        lock (_lock)
        {
            if (!_backends.TryGetValue(_defaultBackendName, out var backend))
            {
                throw new InvalidOperationException($"Default backend '{_defaultBackendName}' not registered");
            }
            return backend;
        }
    }

    public void SetDefaultBackend(string backendName)
    {
        lock (_lock)
        {
            if (!_backends.ContainsKey(backendName))
            {
                throw new InvalidOperationException($"Backend '{backendName}' is not registered");
            }
            _defaultBackendName = backendName;
        }
    }

    public IReadOnlyList<IFusionBackend> GetAllBackends()
    {
        lock (_lock)
        {
            return _backends.Values.ToList();
        }
    }

    public IFusionBackend? FindCapableBackend(IReadOnlyList<Operation> operations)
    {
        lock (_lock)
        {
            foreach (var backend in _backends.Values)
            {
                if (backend.CanFuse(operations))
                {
                    return backend;
                }
            }
        }

        return null;
    }
}
```

### 3. Triton Backend
Implementation for Triton backend.

```csharp
public class TritonBackend : IFusionBackend
{
    private readonly ITritonCompiler _compiler;
    private readonly ITritonAutotuner _autotuner;
    private readonly ILogger _logger;
    private BackendConfiguration? _config;

    public string Name => "Triton";
    public FusionBackendType Type => FusionBackendType.Triton;

    public TritonBackend(
        ITritonCompiler compiler,
        ITritonAutotuner autotuner,
        ILogger logger)
    {
        _compiler = compiler;
        _autotuner = autotuner;
        _logger = logger;
    }

    public void Initialize(BackendConfiguration config)
    {
        _config = config;
        _logger.LogInformation("Initializing Triton backend with device {DeviceId}", config.DeviceId);
    }

    public bool CanFuse(IReadOnlyList<Operation> operations)
    {
        // Triton can fuse most operations
        return GetCapabilities().SupportedPatterns.Overlaps(GetPatternTypes(operations));
    }

    public FusionResult Fuse(IReadOnlyList<Operation> operations, FusionOptions options)
    {
        if (!CanFuse(operations))
        {
            throw new InvalidOperationException("Cannot fuse these operations with Triton");
        }

        // Create fused operation
        var fusedOp = CreateTritonFusedOperation(operations, options);

        // Apply Triton-specific optimizations
        fusedOp = _autotuner.Tune(fusedOp);

        return new FusionResult
        {
            FusedGraph = CreateGraphWithFusedOp(fusedOp),
            FusedOperations = new[] { fusedOp },
            OriginalOpCount = operations.Count,
            FusedOpCount = 1,
            RejectedFusions = Array.Empty<FusionRejected>()
        };
    }

    public CompiledKernel Compile(FusedOperation fusedOp, CompilationOptions options)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        // Compile Triton kernel
        var binary = _compiler.Compile(fusedOp, options);

        stopwatch.Stop();

        // Compute launch config
        var launchConfig = _autotuner.GetBestLaunchConfig(fusedOp);

        return new CompiledKernel
        {
            KernelId = $"{fusedOp.Id}_{Guid.NewGuid():N}",
            Operation = fusedOp,
            Binary = binary,
            LaunchConfig = launchConfig,
            Metrics = new CompilationMetrics
            {
                CompilationTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                BinarySizeBytes = binary.Data.Length,
                OptimizationLevel = options.OptimizationLevel
            }
        };
    }

    public BackendCapabilities GetCapabilities()
    {
        return new BackendCapabilities
        {
            SupportedPatterns = new HashSet<FusionPatternType>
            {
                FusionPatternType.ElementWise,
                FusionPatternType.ConvActivation,
                FusionPatternType.ConvBatchNorm,
                FusionPatternType.LinearActivation,
                FusionPatternType.ReductionThenElementWise
            },
            SupportedDataTypes = new HashSet<TensorDataType>
            {
                TensorDataType.Float32,
                TensorDataType.Float16,
                TensorDataType.BFloat16,
                TensorDataType.Int32,
                TensorDataType.Int64
            },
            SupportsAutotuning = true,
            SupportsProfiling = true,
            SupportsJITCompilation = true,
            SupportsBinaryCache = true
        };
    }

    private FusedOperation CreateTritonFusedOperation(
        IReadOnlyList<Operation> operations,
        FusionOptions options)
    {
        // Create Triton-specific IR
        var ir = BuildTritonIR(operations);

        return new FusedOperation(
            id: Guid.NewGuid().ToString(),
            operations: operations,
            pattern: new FusionPatternDefinition
            {
                Name = "TritonFusion",
                OpTypeSequence = operations.Select(op => op.Type).ToList(),
                MatchStrategy = _ => true,
                Strategy = FusionStrategy.Merge,
                Priority = 10
            },
            ir: ir,
            kernelSpec: new KernelSpecification
            {
                KernelName = $"triton_{ir.Id}",
                Strategy = FusionStrategy.Merge,
                InputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Input).ToList(),
                OutputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Output).ToList(),
                TemporaryMemoryBytes = ir.MemoryLayout.SharedMemoryBytes,
                RegisterBytes = ir.MemoryLayout.RegisterBytes,
                ThreadBlockConfig = new ThreadBlockConfiguration { X = 128, Y = 1, Z = 1 },
                CompilationFlags = new[] { "--fast", "--vectorize" }
            });
    }

    private FusionIR BuildTritonIR(IReadOnlyList<Operation> operations)
    {
        // Build Triton-specific IR
        // Simplified implementation
        var nodes = new List<FusionOpNode>();
        var variables = new List<FusionVariable>();

        string currentVar = "input";
        int nodeCounter = 0;

        foreach (var op in operations)
        {
            var outputVar = $"v_{nodeCounter}";
            nodes.Add(new FusionOpNode
            {
                Id = $"n_{nodeCounter}",
                OriginalOpType = op.Type,
                InputVars = new[] { currentVar },
                OutputVar = outputVar,
                Attributes = op.Attributes.ToDictionary()
            });

            variables.Add(new FusionVariable
            {
                Name = outputVar,
                Shape = op.OutputShape,
                DataType = op.DataType,
                Location = nodeCounter == operations.Count - 1
                    ? MemoryLocation.Output
                    : MemoryLocation.Temporary
            });

            currentVar = outputVar;
            nodeCounter++;
        }

        return new FusionIR
        {
            Id = Guid.NewGuid().ToString("N").Substring(0, 8),
            Nodes = nodes,
            Variables = variables,
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = operations[0].Layout,
                SharedMemoryBytes = 0,
                RegisterBytes = 128
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = 1,
                ThreadsPerBlock = 128,
                RequiresSharedMemory = false,
                RequiresAtomicOps = false
            }
        };
    }

    private ComputationalGraph CreateGraphWithFusedOp(FusedOperation fusedOp)
    {
        // Create a new graph with the fused operation
        // Simplified implementation
        return new ComputationalGraph
        {
            Operations = new List<Operation> { fusedOp },
            Inputs = fusedOp.ConstituentOperations.First().Inputs,
            Outputs = fusedOp.ConstituentOperations.Last().Outputs
        };
    }

    private HashSet<FusionPatternType> GetPatternTypes(IReadOnlyList<Operation> operations)
    {
        // Determine pattern types from operations
        // Simplified implementation
        return new HashSet<FusionPatternType> { FusionPatternType.ElementWise };
    }
}

// Triton-specific interfaces
public interface ITritonCompiler
{
    KernelBinary Compile(FusedOperation fusedOp, CompilationOptions options);
}

public interface ITritonAutotuner
{
    FusedOperation Tune(FusedOperation fusedOp);
    KernelLaunchConfiguration GetBestLaunchConfig(FusedOperation fusedOp);
}
```

### 4. XLA Backend
Implementation for XLA backend.

```csharp
public class XLABackend : IFusionBackend
{
    private readonly IXLACompiler _compiler;
    private readonly ILogger _logger;
    private BackendConfiguration? _config;

    public string Name => "XLA";
    public FusionBackendType Type => FusionBackendType.XLA;

    public XLABackend(IXLACompiler compiler, ILogger logger)
    {
        _compiler = compiler;
        _logger = logger;
    }

    public void Initialize(BackendConfiguration config)
    {
        _config = config;
        _logger.LogInformation("Initializing XLA backend with device {DeviceId}", config.DeviceId);
    }

    public bool CanFuse(IReadOnlyList<Operation> operations)
    {
        // XLA can fuse complex patterns
        return operations.Count <= 20; // XLA has limits on fusion size
    }

    public FusionResult Fuse(IReadOnlyList<Operation> operations, FusionOptions options)
    {
        if (!CanFuse(operations))
        {
            throw new InvalidOperationException("Cannot fuse these operations with XLA");
        }

        // XLA performs automatic fusion during compilation
        // We just need to prepare the operations for XLA
        var fusedOp = CreateXLAFusedOperation(operations);

        return new FusionResult
        {
            FusedGraph = CreateGraphWithFusedOp(fusedOp),
            FusedOperations = new[] { fusedOp },
            OriginalOpCount = operations.Count,
            FusedOpCount = 1,
            RejectedFusions = Array.Empty<FusionRejected>()
        };
    }

    public CompiledKernel Compile(FusedOperation fusedOp, CompilationOptions options)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        // Compile with XLA
        var binary = _compiler.CompileXLAFused(fusedOp, options);

        stopwatch.Stop();

        return new CompiledKernel
        {
            KernelId = $"{fusedOp.Id}_xla",
            Operation = fusedOp,
            Binary = binary,
            LaunchConfig = ComputeXLALaunchConfig(fusedOp),
            Metrics = new CompilationMetrics
            {
                CompilationTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                BinarySizeBytes = binary.Data.Length,
                OptimizationLevel = options.OptimizationLevel
            }
        };
    }

    public BackendCapabilities GetCapabilities()
    {
        return new BackendCapabilities
        {
            SupportedPatterns = new HashSet<FusionPatternType>
            {
                FusionPatternType.ElementWise,
                FusionPatternType.ConvActivation,
                FusionPatternType.ConvBatchNorm,
                FusionPatternType.LinearActivation,
                FusionPatternType.ReductionThenElementWise,
                FusionPatternType.Mixed
            },
            SupportedDataTypes = new HashSet<TensorDataType>
            {
                TensorDataType.Float32,
                TensorDataType.Float16,
                TensorDataType.BFloat16
            },
            SupportsAutotuning = true,
            SupportsProfiling = true,
            SupportsJITCompilation = true,
            SupportsBinaryCache = true
        };
    }

    private FusedOperation CreateXLAFusedOperation(IReadOnlyList<Operation> operations)
    {
        var ir = BuildXLAIR(operations);

        return new FusedOperation(
            id: Guid.NewGuid().ToString(),
            operations: operations,
            pattern: new FusionPatternDefinition
            {
                Name = "XLAFusion",
                OpTypeSequence = operations.Select(op => op.Type).ToList(),
                MatchStrategy = _ => true,
                Strategy = FusionStrategy.Merge,
                Priority = 5
            },
            ir: ir,
            kernelSpec: new KernelSpecification
            {
                KernelName = $"xla_{ir.Id}",
                Strategy = FusionStrategy.Merge,
                InputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Input).ToList(),
                OutputTensors = ir.Variables.Where(v => v.Location == MemoryLocation.Output).ToList(),
                TemporaryMemoryBytes = 0,
                RegisterBytes = 256,
                ThreadBlockConfig = new ThreadBlockConfiguration { X = 256, Y = 1, Z = 1 },
                CompilationFlags = Array.Empty<string>()
            });
    }

    private FusionIR BuildXLAIR(IReadOnlyList<Operation> operations)
    {
        // Build XLA-specific IR
        // Simplified implementation similar to Triton
        var nodes = operations.Select((op, i) => new FusionOpNode
        {
            Id = $"xla_n_{i}",
            OriginalOpType = op.Type,
            InputVars = new[] { i == 0 ? "input" : $"xla_v_{i - 1}" },
            OutputVar = $"xla_v_{i}",
            Attributes = op.Attributes.ToDictionary()
        }).ToList();

        var variables = operations.Select((op, i) => new FusionVariable
        {
            Name = $"xla_v_{i}",
            Shape = op.OutputShape,
            DataType = op.DataType,
            Location = i == operations.Count - 1
                ? MemoryLocation.Output
                : MemoryLocation.Temporary
        }).ToList();

        return new FusionIR
        {
            Id = Guid.NewGuid().ToString("N").Substring(0, 8),
            Nodes = nodes,
            Variables = variables,
            MemoryLayout = new MemoryLayout
            {
                TensorLayout = operations[0].Layout,
                SharedMemoryBytes = 0,
                RegisterBytes = 256
            },
            ComputeRequirements = new ComputeRequirements
            {
                ThreadBlocks = 1,
                ThreadsPerBlock = 256,
                RequiresSharedMemory = false,
                RequiresAtomicOps = false
            }
        };
    }

    private KernelLaunchConfiguration ComputeXLALaunchConfig(FusedOperation fusedOp)
    {
        return new KernelLaunchConfiguration
        {
            BlockDim = new ThreadBlockConfiguration { X = 256, Y = 1, Z = 1 },
            GridDim = new ThreadBlockConfiguration { X = 1, Y = 1, Z = 1 },
            SharedMemoryBytes = 0,
            Parameters = fusedOp.KernelSpec.Parameters.Select(p =>
                new KernelLaunchParameter
                {
                    Name = p.Name,
                    Value = null,
                    Type = p.Type
                }).ToList()
        };
    }

    private ComputationalGraph CreateGraphWithFusedOp(FusedOperation fusedOp)
    {
        return new ComputationalGraph
        {
            Operations = new List<Operation> { fusedOp },
            Inputs = fusedOp.ConstituentOperations.First().Inputs,
            Outputs = fusedOp.ConstituentOperations.Last().Outputs
        };
    }
}

// XLA-specific interfaces
public interface IXLACompiler
{
    KernelBinary CompileXLAFused(FusedOperation fusedOp, CompilationOptions options);
}
```

### 5. Backend Factory
Factory for creating backends.

```csharp
public static class FusionBackendFactory
{
    private static readonly Dictionary<FusionBackendType, Func<IFusionBackend>> _creators = new();

    static FusionBackendFactory()
    {
        RegisterDefaults();
    }

    public static void RegisterBackend(FusionBackendType type, Func<IFusionBackend> creator)
    {
        _creators[type] = creator;
    }

    public static IFusionBackend CreateBackend(FusionBackendType type, BackendConfiguration config)
    {
        if (!_creators.TryGetValue(type, out var creator))
        {
            throw new ArgumentException($"Backend type {type} is not registered", nameof(type));
        }

        var backend = creator();
        backend.Initialize(config);
        return backend;
    }

    public static void RegisterDefaults()
    {
        // Register Triton backend
        RegisterBackend(FusionBackendType.Triton, () =>
            new TritonBackend(
                new TritonCompiler(),
                new TritonAutotuner(),
                new ConsoleLogger()));

        // Register XLA backend
        RegisterBackend(FusionBackendType.XLA, () =>
            new XLABackend(
                new XLACompiler(),
                new ConsoleLogger()));
    }
}
```

## Implementation Tasks

1. **Create backend interfaces** (20 min)
   - IFusionBackend interface
   - IFusionBackendRegistry interface
   - Backend-related records and enums
   - BackendCapabilities record

2. **Implement FusionBackendRegistry** (25 min)
   - Registry implementation
   - Thread-safe backend management
   - Default backend handling
   - Capable backend lookup

3. **Implement TritonBackend** (35 min)
   - Backend implementation
   - CanFuse logic
   - Fuse operation creation
   - Compile method
   - GetCapabilities method
   - Triton-specific interfaces

4. **Implement XLABackend** (30 min)
   - Backend implementation
   - CanFuse logic
   - Fuse operation creation
   - Compile method
   - GetCapabilities method
   - XLA-specific interfaces

5. **Implement FusionBackendFactory** (15 min)
   - Factory implementation
   - Backend registration
   - Default backend registration

## Test Cases

```csharp
[Test]
public void Registry_RegisterAndRetrieveBackend()
{
    var registry = new FusionBackendRegistry();
    var backend = CreateMockBackend();

    registry.Register(backend);
    var retrieved = registry.GetBackend(backend.Name);

    Assert.IsNotNull(retrieved);
    Assert.AreEqual(backend.Name, retrieved.Name);
}

[Test]
public void Registry_SetDefaultBackend()
{
    var registry = new FusionBackendRegistry();
    var backend1 = CreateMockBackend("Backend1");
    var backend2 = CreateMockBackend("Backend2");

    registry.Register(backend1);
    registry.Register(backend2);

    registry.SetDefaultBackend("Backend2");
    var defaultBackend = registry.GetDefaultBackend();

    Assert.AreEqual("Backend2", defaultBackend.Name);
}

[Test]
public void TritonBackend_CanFuseElementWise()
{
    var backend = new TritonBackend(new MockCompiler(), new MockAutotuner(), new ConsoleLogger());
    var operations = new[] { CreateOperation("Add"), CreateOperation("Mul") };

    Assert.IsTrue(backend.CanFuse(operations));
}

[Test]
public void XLABackend_GetCapabilities()
{
    var backend = new XLABackend(new MockXLACompiler(), new ConsoleLogger());
    var capabilities = backend.GetCapabilities();

    Assert.IsTrue(capabilities.SupportsAutotuning);
    Assert.IsTrue(capabilities.SupportsJITCompilation);
    Assert.Contains(FusionPatternType.ConvActivation, capabilities.SupportedPatterns);
}
```

## Success Criteria
- Backend interface provides common abstraction
- Registry correctly manages multiple backends
- Triton backend implements interface correctly
- XLA backend implements interface correctly
- Factory creates backends with proper initialization
- Capabilities are accurately reported

## Dependencies
- FusedOperation from fusion engine
- FusionOptions and CompilationOptions
- Operation abstraction
- ILogger interface
- Triton and XLA compiler interfaces (to be implemented later)
