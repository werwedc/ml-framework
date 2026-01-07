# Spec: Just-In-Time Compilation (jit)

## Overview
Implement the JIT compilation transformation that traces function execution and compiles to optimized kernels.

## Scope
- Implement Compile method
- Trace function execution
- Basic optimization (operator fusion placeholder)
- Caching of compiled functions

## Technical Requirements

### 1. JIT Transform Implementation

```csharp
namespace MLFramework.Functional.Compilation
{
    public class JITTransform : BaseTransformation
    {
        private static readonly ConcurrentDictionary<Delegate, CompiledFunction> _compiledCache =
            new ConcurrentDictionary<Delegate, CompiledFunction>();

        public JITTransform(Delegate original)
            : base("jit", TransformationType.Compilation)
        {
            ValidateDelegate(original);
        }

        public override Delegate Transform(Delegate original)
        {
            // Check cache first
            if (_compiledCache.TryGetValue(original, out var compiled))
            {
                return compiled.AsDelegate();
            }

            // Trace and compile the function
            var traceContext = TraceAndCompile(original);
            compiled = new CompiledFunction(original, traceContext);

            // Cache the result
            _compiledCache.TryAdd(original, compiled);

            return compiled.AsDelegate();
        }

        private TraceContext TraceAndCompile(Delegate original)
        {
            var method = original.Method;
            var returnType = method.ReturnType;

            // For now, only support Func<Tensor, Tensor> and Func<Tensor, Tensor, Tensor>
            if (method.GetParameters().Length == 1 &&
                method.GetParameters()[0].ParameterType == typeof(Tensor))
            {
                return TraceSingleInput((Func<Tensor, Tensor>)original);
            }

            if (method.GetParameters().Length == 2 &&
                method.GetParameters()[0].ParameterType == typeof(Tensor) &&
                method.GetParameters()[1].ParameterType == typeof(Tensor))
            {
                return TraceDoubleInput((Func<Tensor, Tensor, Tensor>)original);
            }

            throw new NotSupportedException("Unsupported delegate signature for JIT compilation");
        }

        private TraceContext TraceSingleInput(Func<Tensor, Tensor> original)
        {
            using (var trace = new TraceContext())
            {
                // Create a wrapper that traces execution
                Func<Tensor, Tensor> tracedWrapper = (Tensor input) =>
                {
                    // Convert input to TracedTensor
                    var tracedInput = TracedTensor.Create(input, "input");

                    // Execute with traced tensors
                    // Note: This requires the function to work with TracedTensor
                    // For now, we'll use a different approach

                    throw new NotImplementedException("Function must be marked with [TensorFunction] attribute");
                };

                return trace;
            }
        }

        private TraceContext TraceDoubleInput(Func<Tensor, Tensor, Tensor> original)
        {
            using (var trace = new TraceContext())
            {
                // Similar to single input
                return trace;
            }
        }
    }
}
```

### 2. CompiledFunction Class

```csharp
/// <summary>
/// Represents a compiled function with cached execution plan.
/// </summary>
public class CompiledFunction
{
    private readonly Delegate _original;
    private readonly TraceContext _trace;
    private readonly Func<Delegate, Delegate> _compiledDelegateFactory;

    public CompiledFunction(Delegate original, TraceContext trace)
    {
        _original = original ?? throw new ArgumentNullException(nameof(original));
        _trace = trace ?? throw new ArgumentNullException(nameof(trace));

        // Create compiled delegate factory based on original signature
        _compiledDelegateFactory = CreateCompiledDelegateFactory();
    }

    private Func<Delegate, Delegate> CreateCompiledDelegateFactory()
    {
        var method = _original.Method;

        if (method.GetParameters().Length == 1 &&
            method.GetParameters()[0].ParameterType == typeof(Tensor))
        {
            return CreateSingleInputCompiledFactory();
        }

        if (method.GetParameters().Length == 2 &&
            method.GetParameters()[0].ParameterType == typeof(Tensor) &&
            method.GetParameters()[1].ParameterType == typeof(Tensor))
        {
            return CreateDoubleInputCompiledFactory();
        }

        throw new NotSupportedException("Unsupported delegate signature");
    }

    private Func<Delegate, Delegate> CreateSingleInputCompiledFactory()
    {
        return original =>
        {
            var func = (Func<Tensor, Tensor>)original;
            return (Tensor input) =>
            {
                // For now, just execute the original function
                // In a full implementation, this would use the compiled kernel
                return func(input);
            };
        };
    }

    private Func<Delegate, Delegate> CreateDoubleInputCompiledFactory()
    {
        return original =>
        {
            var func = (Func<Tensor, Tensor, Tensor>)original;
            return (Tensor input1, Tensor input2) =>
            {
                // For now, just execute the original function
                return func(input1, input2);
            };
        };
    }

    public Delegate AsDelegate()
    {
        return _compiledDelegateFactory(_original);
    }
}
```

### 3. Compile Extension Method

```csharp
namespace MLFramework.Functional
{
    public static class Functional
    {
        /// <summary>
        /// Just-in-time compile a function for optimization.
        /// </summary>
        public static Func<Tensor, Tensor> Compile(Func<Tensor, Tensor> func)
        {
            var transform = new JITTransform(func);
            return (Func<Tensor, Tensor>)transform.Transform(func);
        }

        public static Func<Tensor, Tensor, Tensor> Compile(Func<Tensor, Tensor, Tensor> func)
        {
            var transform = new JITTransform(func);
            return (Func<Tensor, Tensor, Tensor>)transform.Transform(func);
        }

        /// <summary>
        /// Clear the JIT compilation cache.
        /// </summary>
        public static void ClearJITCache()
        {
            JITTransform.ClearCache();
        }
    }
}
```

### 4. Add Cache Management to JITTransform

```csharp
public class JITTransform : BaseTransformation
{
    private static readonly ConcurrentDictionary<Delegate, CompiledFunction> _compiledCache =
        new ConcurrentDictionary<Delegate, CompiledFunction>();

    // ... existing code ...

    public static void ClearCache()
    {
        _compiledCache.Clear();
    }

    public static int CacheSize => _compiledCache.Count;
}
```

### 5. Simplified Trace-and-Execute Approach

Since full automatic tracing is complex, implement a simpler approach using attribute-based tracing:

```csharp
[AttributeUsage(AttributeTargets.Method)]
public class JITTraceableAttribute : Attribute
{
    public bool ForceCompilation { get; set; }
}

public static class JITTracer
{
    public static TraceContext Trace<TDelegate>(TDelegate func, params Tensor[] exampleInputs)
        where TDelegate : Delegate
    {
        using (var trace = new TraceContext())
        {
            // Create traced versions of inputs
            var tracedInputs = exampleInputs.Select((t, i) =>
                TracedTensor.Create(t, $"input_{i}")).ToArray();

            // Execute function with traced inputs
            // This requires the function to be implemented in a traceable way
            // For now, this is a placeholder for the tracing mechanism

            return trace;
        }
    }
}
```

## Files to Create
1. `src/MLFramework/Functional/Compilation/JITTransform.cs`
2. `src/MLFramework/Functional/Compilation/CompiledFunction.cs`
3. `src/MLFramework/Functional/Compilation/JITTraceableAttribute.cs`
4. `src/MLFramework/Functional/Compilation/JITTracer.cs`
5. Update `src/MLFramework/Functional/Functional.cs` with Compile methods

## Dependencies
- spec_functional_core_interfaces.md
- spec_functional_tracing.md (must be completed first)

## Success Criteria
- Can wrap functions with Compile()
- Basic caching works
- Returns compiled delegate
- Can clear cache
- Supports Func<Tensor, Tensor> and Func<Tensor, Tensor, Tensor>

## Notes for Coder
- This is a basic JIT implementation
- Actual compilation to machine code is complex - use placeholders for now
- Focus on the infrastructure: caching, delegation, trace management
- The "compilation" step can just return the original function for now
- Real compilation (lowering to IR, optimization) is future work
- Make sure cache is thread-safe (ConcurrentDictionary)

## Example Usage
```csharp
// Define a function
Tensor ModelForward(Tensor input)
{
    var x = DenseLayer1(input);
    x = ReLU(x);
    x = DenseLayer2(x);
    return x;
}

// Compile it
var compiled = Functional.Compile(ModelForward);

// Use compiled version (should be faster in real implementation)
var result = compiled(input);
```

## Future Enhancements
- Actual compilation to optimized kernels
- Shape and type inference for better optimization
- Control flow handling
- Operator fusion
- Memory optimization
