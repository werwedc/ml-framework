# Spec: AMP Operation Precision Registry

## Overview
Create a registry system for defining which operations should run in which precision (FP16, BF16, or FP32) during automatic mixed precision training.

## Class Specification

### 1. OpPrecision Enum

**File:** `src/MLFramework/Amp/OpPrecision.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Precision policy for operations in AMP
    /// </summary>
    public enum OpPrecision
    {
        /// <summary>
        /// Use higher precision (FP32)
        /// </summary>
        Higher = 0,

        /// <summary>
        /// Use lower precision (FP16/BF16 based on config)
        /// </summary>
        Lower = 1,

        /// <summary>
        /// Keep original precision
        /// </summary>
        Keep = 2,

        /// <summary>
        /// Custom precision specified separately
        /// </summary>
        Custom = 3
    }
}
```

### 2. OpPrecisionRule Class

**File:** `src/MLFramework/Amp/OpPrecisionRule.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Defines precision rules for a specific operation
    /// </summary>
    public class OpPrecisionRule
    {
        /// <summary>
        /// Gets the operation type this rule applies to
        /// </summary>
        public Type OperationType { get; }

        /// <summary>
        /// Gets the forward pass precision policy
        /// </summary>
        public OpPrecision ForwardPrecision { get; }

        /// <summary>
        /// Gets the backward pass precision policy
        /// </summary>
        public OpPrecision BackwardPrecision { get; }

        /// <summary>
        /// Gets or sets custom forward dtype (if ForwardPrecision is Custom)
        /// </summary>
        public DataType? CustomForwardDtype { get; set; }

        /// <summary>
        /// Gets or sets custom backward dtype (if BackwardPrecision is Custom)
        /// </summary>
        public DataType? CustomBackwardDtype { get; set; }

        /// <summary>
        /// Gets or sets the priority (higher = more important)
        /// </summary>
        public int Priority { get; set; }

        /// <summary>
        /// Creates a new OpPrecisionRule
        /// </summary>
        public OpPrecisionRule(
            Type operationType,
            OpPrecision forwardPrecision,
            OpPrecision backwardPrecision = OpPrecision.Keep,
            int priority = 0);

        /// <summary>
        /// Gets the actual forward dtype based on AMP config
        /// </summary>
        public DataType GetForwardDtype(AmpConfig config);

        /// <summary>
        /// Gets the actual backward dtype based on AMP config
        /// </summary>
        public DataType GetBackwardDtype(AmpConfig config);
    }
}
```

### 3. AmpRegistry Class

**File:** `src/MLFramework/Amp/AmpRegistry.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Registry for operation-specific precision rules in AMP
    /// </summary>
    public class AmpRegistry
    {
        private readonly Dictionary<Type, OpPrecisionRule> _rules;
        private readonly AmpConfig _config;
        private readonly object _lock = new object();

        /// <summary>
        /// Creates a new AmpRegistry with default rules
        /// </summary>
        public AmpRegistry(AmpConfig config);

        /// <summary>
        /// Registers an operation to the whitelist (use lower precision)
        /// </summary>
        public void RegisterWhitelist(Type operationType, int priority = 0);

        /// <summary>
        /// Registers an operation to the blacklist (use higher precision)
        /// </summary>
        public void RegisterBlacklist(Type operationType, int priority = 0);

        /// <summary>
        /// Registers a custom precision rule for an operation
        /// </summary>
        public void RegisterCustomOp(
            Type operationType,
            DataType forwardDtype,
            DataType backwardDtype,
            int priority = 0);

        /// <summary>
        /// Registers a full precision rule for an operation
        /// </summary>
        public void RegisterRule(OpPrecisionRule rule);

        /// <summary>
        /// Removes a rule for an operation
        /// </summary>
        public void Unregister(Type operationType);

        /// <summary>
        /// Gets the precision rule for an operation
        /// </summary>
        public OpPrecisionRule GetRule(Type operationType);

        /// <summary>
        /// Gets the forward dtype for an operation
        /// </summary>
        public DataType GetForwardDtype(Type operationType, DataType inputDtype);

        /// <summary>
        /// Gets the backward dtype for an operation
        /// </summary>
        public DataType GetBackwardDtype(Type operationType, DataType inputDtype);

        /// <summary>
        /// Checks if an operation is in the whitelist
        /// </summary>
        public bool IsWhitelisted(Type operationType);

        /// <summary>
        /// Checks if an operation is in the blacklist
        /// </summary>
        public bool IsBlacklisted(Type operationType);

        /// <summary>
        /// Clears all registered rules
        /// </summary>
        public void Clear();

        /// <summary>
        /// Gets all registered rules
        /// </summary>
        public IReadOnlyDictionary<Type, OpPrecisionRule> GetAllRules();
    }
}
```

### 4. DefaultAmpRules Class

**File:** `src/MLFramework/Amp/DefaultAmpRules.cs`

```csharp
namespace MLFramework.Amp
{
    /// <summary>
    /// Default precision rules for common operations
    /// </summary>
    public static class DefaultAmpRules
    {
        /// <summary>
        /// Gets the default whitelist (operations safe for FP16/BF16)
        /// </summary>
        public static Type[] DefaultWhitelist => new Type[]
        {
            typeof(Conv2d),
            typeof(Conv3d),
            typeof(Linear),
            typeof(MaxPool2d),
            typeof(MaxPool3d),
            typeof(AvgPool2d),
            typeof(AvgPool3d),
            typeof(BatchNorm1d),
            typeof(BatchNorm2d),
            typeof(BatchNorm3d),
            typeof(Dropout),
            typeof(Relu),
            typeof(Gelu),
            typeof(Sigmoid),
            typeof(Tanh),
            typeof(ElementwiseAdd),
            typeof(ElementwiseMul),
            typeof(ElementwiseSub),
            typeof(ElementwiseDiv),
            typeof(MatMul),
            typeof(MatrixMultiply)
        };

        /// <summary>
        /// Gets the default blacklist (operations requiring FP32)
        /// </summary>
        public static Type[] DefaultBlacklist => new Type[]
        {
            typeof(Softmax),
            typeof(LogSoftmax),
            typeof(Log),
            typeof(Exp),
            typeof(Sqrt),
            typeof(ReduceSum),
            typeof(ReduceMean),
            typeof(ReduceMax),
            typeof(ReduceMin),
            typeof(Normalize),
            typeof(LayerNorm),
            typeof(Embedding),
            typeof(CrossEntropyLoss),
            typeof(NllLoss),
            typeof(KlDivLoss),
            typeof(PoissonNLLLoss)
        };

        /// <summary>
        /// Applies default rules to the registry
        /// </summary>
        public static void ApplyDefaultRules(AmpRegistry registry);
    }
}
```

## Implementation Details

### Registry Lookup Logic
1. Check if operation type has a registered rule
2. If no rule, apply default policy:
   - Matrix operations (conv, linear) -> Lower precision
   - Reductions (sum, mean) -> Higher precision
   - Elementwise (add, mul) -> Keep input precision
3. Return the appropriate dtype based on config

### Priority System
- Higher priority rules override lower priority rules
- Default rules have priority 0
- User-defined rules typically have priority > 0
- Latest registration wins for same priority

### Thread Safety
- All registry operations use locking
- Multiple threads can read simultaneously
- Write operations are mutually exclusive

## Default Rules Reference

### FP16/BF16 Safe (Whitelist)
- Convolutions (Conv2d, Conv3d)
- Linear layers
- Pooling operations
- Batch normalization
- Dropout
- Activation functions (ReLU, GELU, Sigmoid, Tanh)
- Elementwise operations
- Matrix multiplication

### FP32 Required (Blacklist)
- Softmax, LogSoftmax
- Logarithms, Exponentials
- Square root
- Reduction operations (sum, mean, max, min)
- Layer normalization
- Embedding lookup
- Loss functions

## Dependencies
- MLFramework.Core (DataType, Type)
- System.Collections.Concurrent (for thread safety)
- System.Reflection (for operation type checking)

## Testing Requirements
- Test registration of whitelist operations
- Test registration of blacklist operations
- Test custom precision rules
- Test priority-based rule override
- Test thread safety with concurrent access
- Test default rule application
- Test lookup with unknown operations

## Success Criteria
- [ ] Registry correctly stores and retrieves rules
- [ ] Default rules are applied correctly
- [ ] Custom rules override default rules
- [ ] Priority system works as expected
- [ ] Thread-safe operation verified
- [ ] All unit tests pass
- [ ] Documentation of default rules complete
