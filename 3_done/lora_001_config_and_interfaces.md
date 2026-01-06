# Spec: LoRA Configuration and Interfaces

## Overview
Define the core configuration and interfaces for LoRA (Low-Rank Adaptation) support in the ML Framework. This spec establishes the foundational types that all other LoRA components will depend on.

## Implementation Details

### 1. LoRA Initialization Strategies
**File**: `src/LoRA/LoRAInitializationStrategy.cs`

Define an enum for adapter weight initialization:

```csharp
public enum LoRAInitializationStrategy
{
    /// <summary>
    /// Initialize A with Kaiming normal, B with zeros (standard LoRA approach)
    /// </summary>
    Standard,

    /// <summary>
    /// Initialize both matrices with Xavier uniform
    /// </summary>
    Xavier,

    /// <summary>
    /// Initialize all weights to zero (start with zero perturbation)
    /// </summary>
    Zero
}
```

### 2. LoRA Configuration
**File**: `src/LoRA/LoRAConfig.cs`

Define the configuration class for LoRA adapters:

```csharp
public class LoRAConfig
{
    /// <summary>
    /// Rank of the low-rank decomposition (typically 4-64)
    /// </summary>
    public int Rank { get; set; } = 8;

    /// <summary>
    /// Scaling factor alpha (controls adapter influence)
    /// </summary>
    public float Alpha { get; set; } = 16.0f;

    /// <summary>
    /// Target modules to apply LoRA to (e.g., ["attn.q_proj", "attn.v_proj"])
    /// If null/empty, applies to all compatible layers
    /// </summary>
    public string[]? TargetModules { get; set; } = null;

    /// <summary>
    /// Whether to apply LoRA to bias terms (default: false)
    /// </summary>
    public bool UseBias { get; set; } = false;

    /// <summary>
    /// Initialization strategy for adapter weights
    /// </summary>
    public LoRAInitializationStrategy Initialization { get; set; } = LoRAInitializationStrategy.Standard;

    /// <summary>
    /// Dropout rate for LoRA layers (0.0 = no dropout)
    /// </summary>
    public float Dropout { get; set; } = 0.0f;

    /// <summary>
    /// Whether to use fused kernels if available (performance optimization)
    /// </summary>
    public bool UseFusedKernels { get; set; } = false;

    /// <summary>
    /// Target parameter types (Linear, Conv2d, Embedding, etc.)
    /// If null/empty, applies to all types
    /// </summary>
    public string[]? TargetLayerTypes { get; set; } = null;

    public LoRAConfig(int rank = 8, float alpha = 16.0f)
    {
        Rank = rank;
        Alpha = alpha;
        Validate();
    }

    private void Validate()
    {
        if (Rank <= 0)
            throw new ArgumentException("Rank must be positive", nameof(Rank));
        if (Alpha <= 0)
            throw new ArgumentException("Alpha must be positive", nameof(Alpha));
        if (Dropout < 0 || Dropout >= 1)
            throw new ArgumentException("Dropout must be in [0, 1)", nameof(Dropout));
    }
}
```

### 3. ILoRAAdapter Interface
**File**: `src/LoRA/ILoRAAdapter.cs`

Define the core interface that all LoRA wrapper layers must implement:

```csharp
public interface ILoRAAdapter
{
    /// <summary>
    /// Gets the base (wrapped) layer
    /// </summary>
    IModule BaseLayer { get; }

    /// <summary>
    /// Gets the LoRA rank
    /// </summary>
    int Rank { get; }

    /// <summary>
    /// Gets the LoRA scaling factor (alpha / rank)
    /// </summary>
    float ScalingFactor { get; }

    /// <summary>
    /// Freezes the base layer (only adapter parameters remain trainable)
    /// </summary>
    void FreezeBaseLayer();

    /// <summary>
    /// Unfreezes the base layer
    /// </summary>
    void UnfreezeBaseLayer();

    /// <summary>
    /// Gets all trainable parameters (adapter weights only if base is frozen)
    /// </summary>
    IEnumerable<ITensor> TrainableParameters { get; }

    /// <summary>
    /// Gets all frozen parameters
    /// </summary>
    IEnumerable<ITensor> FrozenParameters { get; }

    /// <summary>
    /// Enables or disables the LoRA adapter
    /// </summary>
    bool IsEnabled { get; set; }

    /// <summary>
    /// Merges adapter weights into the base layer (for deployment)
    /// </summary>
    void MergeAdapter();

    /// <summary>
    /// Resets the base layer to original weights (undoes merge)
    /// </summary>
    void ResetBaseLayer();

    /// <summary>
    /// Gets the adapter weights as tensors
    /// </summary>
    (ITensor? MatrixA, ITensor? MatrixB) GetAdapterWeights();

    /// <summary>
    /// Sets the adapter weights from tensors
    /// </summary>
    void SetAdapterWeights(ITensor? matrixA, ITensor? matrixB);
}
```

### 4. LoRAAdapterBase Abstract Class
**File**: `src/LoRA/LoRAAdapterBase.cs`

Provide a base implementation with common functionality:

```csharp
public abstract class LoRAAdapterBase : ILoRAAdapter
{
    protected readonly IModule _baseLayer;
    protected readonly int _rank;
    protected readonly float _alpha;
    protected bool _isBaseLayerFrozen;
    protected bool _isEnabled = true;
    protected ITensor? _baseLayerWeightsBackup;

    protected LoRAAdapterBase(IModule baseLayer, int rank, float alpha)
    {
        _baseLayer = baseLayer ?? throw new ArgumentNullException(nameof(baseLayer));
        _rank = rank;
        _alpha = alpha;
        _isBaseLayerFrozen = false;
    }

    public IModule BaseLayer => _baseLayer;
    public int Rank => _rank;
    public float ScalingFactor => _alpha / _rank;
    public bool IsEnabled
    {
        get => _isEnabled;
        set => _isEnabled = value;
    }

    public abstract void FreezeBaseLayer();
    public abstract void UnfreezeBaseLayer();
    public abstract IEnumerable<ITensor> TrainableParameters { get; }
    public abstract IEnumerable<ITensor> FrozenParameters { get; }
    public abstract void MergeAdapter();
    public abstract void ResetBaseLayer();
    public abstract (ITensor? MatrixA, ITensor? MatrixB) GetAdapterWeights();
    public abstract void SetAdapterWeights(ITensor? matrixA, ITensor? matrixB);
}
```

## Testing Requirements

**File**: `tests/LoRA/LoRAConfigTests.cs`

1. Test default configuration values
2. Test configuration validation (invalid rank, alpha, dropout)
3. Test configuration with custom target modules
4. Test different initialization strategies

**File**: `tests/LoRA/LoRAAdapterBaseTests.cs`

1. Test base layer wrapping
2. Test scaling factor calculation
3. Test enable/disable functionality

## Dependencies
- `IModule` interface (existing)
- `ITensor` interface (existing)

## Success Criteria
- Configuration class validates all inputs correctly
- Interface defines all necessary operations
- Base class provides common functionality
- All types compile without errors
- Unit tests pass for configuration validation

## Estimated Time
45 minutes

## Notes
- This spec establishes the foundation for all subsequent LoRA specs
- Ensure interface methods are flexible enough for different layer types (Linear, Conv2d, Embedding)
- Consider extensibility for future features (AdaLoRA, QLoRA)
