# Spec: AMP AutoCast Context

## Overview
Implement an AutoCast context manager that automatically converts tensors to the appropriate precision (FP16/BF16) during the forward pass based on the operation registry.

## Class Specification

### 1. AutoCastMode Enum

**File:** `src/MLFramework/Amp/AutoCastMode.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// AutoCast mode for precision conversion
    /// </summary>
    public enum AutoCastMode
    {
        /// <summary>
        /// Cast to FP16 (Half precision)
        /// </summary>
        Fp16 = 0,

        /// <summary>
        /// Cast to BF16 (Brain Float)
        /// </summary>
        Bf16 = 1,

        /// <summary>
        /// No casting (keep original precision)
        /// </summary>
        None = 2
    }
}
```

### 2. AutoCast Class

**File:** `src/MLFramework/Amp/AutoCast.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Context manager for automatic mixed precision casting
    /// </summary>
    public class AutoCast : IDisposable
    {
        private readonly AutoCastMode _mode;
        private readonly AmpRegistry _registry;
        private readonly bool _enabled;
        private readonly Stack<AutoCast> _contextStack;

        /// <summary>
        /// Gets the current AutoCast mode
        /// </summary>
        public AutoCastMode Mode => _mode;

        /// <summary>
        /// Gets whether AutoCast is enabled
        /// </summary>
        public bool Enabled => _enabled;

        /// <summary>
        /// Gets the current active AutoCast context (thread-local)
        /// </summary>
        public static AutoCast? Current { get; private set; }

        /// <summary>
        /// Creates a new AutoCast context with BF16 mode (recommended)
        /// </summary>
        /// <param name="enabled">Whether to enable AutoCast (default: true)</param>
        /// <param name="registry">The operation precision registry (default: null for default registry)</param>
        public AutoCast(bool enabled = true, AmpRegistry? registry = null);

        /// <summary>
        /// Creates a new AutoCast context with specified mode
        /// </summary>
        /// <param name="mode">The AutoCast mode (FP16, BF16, or None)</param>
        /// <param name="enabled">Whether to enable AutoCast (default: true)</param>
        /// <param name="registry">The operation precision registry (default: null for default registry)</param>
        public AutoCast(AutoCastMode mode, bool enabled = true, AmpRegistry? registry = null);

        /// <summary>
        /// Casts a tensor to the appropriate precision for the current operation
        /// </summary>
        /// <param name="tensor">The tensor to cast</param>
        /// <param name="operationType">The type of operation being performed</param>
        /// <returns>Casted tensor</returns>
        public Tensor Cast(Tensor tensor, Type operationType);

        /// <summary>
        /// Casts a tensor to a specific dtype
        /// </summary>
        /// <param name="tensor">The tensor to cast</param>
        /// <param name="dtype">The target data type</param>
        /// <returns>Casted tensor</returns>
        public Tensor Cast(Tensor tensor, DataType dtype);

        /// <summary>
        /// Gets the forward dtype for an operation
        /// </summary>
        /// <param name="operationType">The type of operation</param>
        /// <param name="inputDtype">The input tensor dtype</param>
        /// <returns>The target dtype for the operation</returns>
        public DataType GetForwardDtype(Type operationType, DataType inputDtype);

        /// <summary>
        /// Gets the backward dtype for an operation
        /// </summary>
        /// <param name="operationType">The type of operation</param>
        /// <param name="inputDtype">The input tensor dtype</param>
        /// <returns>The target dtype for the operation</returns>
        public DataType GetBackwardDtype(Type operationType, DataType inputDtype);

        /// <summary>
        /// Enters the AutoCast context
        /// </summary>
        public void Enter();

        /// <summary>
        /// Exits the AutoCast context
        /// </summary>
        public void Exit();

        /// <summary>
        /// Disposes the context and restores previous context
        /// </summary>
        public void Dispose();
    }
}
```

### 3. AutoCastContext Class

**File:** `src/MLFramework/Amp/AutoCastContext.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Convenience methods for creating AutoCast contexts
    /// </summary>
    public static class AutoCastContext
    {
        /// <summary>
        /// Creates an AutoCast context with FP16 mode
        /// </summary>
        public static AutoCast Fp16(AmpRegistry? registry = null);

        /// <summary>
        /// Creates an AutoCast context with BF16 mode (recommended)
        /// </summary>
        public static AutoCast Bf16(AmpRegistry? registry = null);

        /// <summary>
        /// Creates an AutoCast context with the specified mode
        /// </summary>
        /// <param name="mode">The AutoCast mode</param>
        /// <param name="registry">The operation precision registry</param>
        public static AutoCast Create(AutoCastMode mode, AmpRegistry? registry = null);

        /// <summary>
        /// Checks if AutoCast is currently active
        /// </summary>
        public static bool IsActive { get; }

        /// <summary>
        /// Gets the current AutoCast mode (returns None if not active)
        /// </summary>
        public static AutoCastMode CurrentMode { get; }

        /// <summary>
        /// Casts a tensor to the appropriate precision (uses current context)
        /// </summary>
        /// <param name="tensor">The tensor to cast</param>
        /// <param name="operationType">The type of operation being performed</param>
        /// <returns>Casted tensor</returns>
        public static Tensor Cast(Tensor tensor, Type operationType);

        /// <summary>
        /// Casts a tensor to a specific dtype (uses current context)
        /// </summary>
        /// <param name="tensor">The tensor to cast</param/// <param name="dtype">The target data type</param>
        /// <returns>Casted tensor</returns>
        public static Tensor Cast(Tensor tensor, DataType dtype);
    }
}
```

### 4. AmpEnabled Attribute

**File:** `src/MLFramework/Amp/AmpEnabled.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Attribute to mark methods as AMP-aware
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Class)]
    public class AmpEnabledAttribute : Attribute
    {
        /// <summary>
        /// Gets or sets the default forward precision for this method
        /// </summary>
        public DataType? DefaultForwardDtype { get; set; }

        /// <summary>
        /// Gets or sets the default backward precision for this method
        /// </summary>
        public DataType? DefaultBackwardDtype { get; set; }

        /// <summary>
        /// Creates a new AmpEnabledAttribute
        /// </summary>
        public AmpEnabledAttribute();

        /// <summary>
        /// Creates a new AmpEnabledAttribute with specified precision
        /// </summary>
        /// <param name="forwardDtype">The default forward precision</param>
        /// <param name="backwardDtype">The default backward precision</param>
        public AmpEnabledAttribute(DataType forwardDtype, DataType backwardDtype);
    }
}
```

## Implementation Details

### AutoCast Logic

```csharp
public Tensor Cast(Tensor tensor, Type operationType)
{
    if (!_enabled || _mode == AutoCastMode.None)
    {
        return tensor;
    }

    var targetDtype = GetForwardDtype(operationType, tensor.Dtype);

    // Skip casting if already at target dtype
    if (tensor.Dtype == targetDtype)
    {
        return tensor;
    }

    // Cast to target dtype
    return tensor.Cast(targetDtype);
}

public DataType GetForwardDtype(Type operationType, DataType inputDtype)
{
    var rule = _registry.GetRule(operationType);

    if (rule != null)
    {
        return rule.GetForwardDtype(new AmpConfig
        {
            TargetPrecision = _mode == AutoCastMode.Fp16 ? DataType.Float16 : DataType.BFloat16,
            HigherPrecision = DataType.Float32
        });
    }

    // Default policy: use input dtype
    return inputDtype;
}
```

### Context Management

```csharp
public void Enter()
{
    _contextStack.Push(Current);
    Current = this;
}

public void Exit()
{
    if (_contextStack.Count > 0)
    {
        Current = _contextStack.Pop();
    }
}

public void Dispose()
{
    Exit();
}
```

### Thread-Local Storage

```csharp
// Use AsyncLocal for thread-local context management
private static readonly AsyncLocal<AutoCast?> _current = new AsyncLocal<AutoCast?>();
public static AutoCast? Current
{
    get => _current.Value;
    private set => _current.Value = value;
}
```

## Usage Examples

### Basic Usage
```csharp
using (var autocast = new AutoCast(AutoCastMode.Bf16))
{
    // Forward pass with automatic casting
    var output = model.Forward(inputs); // Casts to BF16 automatically
    var loss = criterion(output, targets);

    loss.Backward();
    optimizer.Step();
}
```

### Using AutoCastContext
```csharp
using (var autocast = AutoCastContext.Bf16())
{
    var output = model.Forward(inputs);
    var loss = criterion(output, targets);
    loss.Backward();
    optimizer.Step();
}
```

### Explicit Casting
```csharp
using (var autocast = AutoCastContext.Bf16())
{
    var x = AutoCastContext.Cast(input, typeof(Conv2d));
    var output = model.Forward(x);
}
```

### Custom Registry
```csharp
var registry = new AmpRegistry(AmpConfig.CreateBf16());
registry.RegisterWhitelist(typeof(MyCustomOp));

using (var autocast = new AutoCast(AutoCastMode.Bf16, registry: registry))
{
    var output = model.Forward(inputs);
}
```

## Performance Considerations

### Zero-Copy Casting
- Use view casting when tensor layout permits
- Avoid unnecessary dtype conversions
- Cache dtype conversion decisions

### Overhead Minimization
- Check if casting is needed before actual cast
- Use cached scale tensors for loss scaling
- Minimize context switching overhead

## Dependencies
- MLFramework.Core (Tensor, DataType)
- MLFramework.Amp (AmpRegistry, AmpConfig)
- System (IDisposable, AsyncLocal)
- System.Collections.Concurrent (for thread safety)

## Testing Requirements
- Test AutoCast context enter/exit
- Test tensor casting for whitelisted operations
- Test tensor casting for blacklisted operations
- Test custom registry with AutoCast
- Test nested AutoCast contexts
- Test AutoCast with different modes (FP16, BF16, None)
- Test AutoCastContext convenience methods
- Test thread-local context isolation
- Test AmpEnabled attribute (if used)

## Success Criteria
- [ ] AutoCast context correctly manages enter/exit
- [ ] Tensors are cast to correct precision based on registry
- [ ] Whitelisted operations use lower precision
- [ ] Blacklisted operations use higher precision
- [ ] Nested contexts work correctly
- [ ] Thread-local isolation is maintained
- [ ] Performance overhead < 5% of forward pass time
- [ ] All unit tests pass
- [ ] Documentation includes usage examples
