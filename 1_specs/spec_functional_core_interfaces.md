# Spec: Core Functional Transformation Interfaces

## Overview
Define the foundational interfaces and base classes for the functional transformation system. This provides the abstraction layer that all transformations (vmap, pmap, jit) will build upon.

## Scope
- Define transformation interfaces
- Create base transformation class
- Define functional utility attributes
- Define transformation context and metadata

## Technical Requirements

### 1. Core Interfaces

```csharp
// Base transformation interface
public interface IFunctionalTransformation
{
    // Apply transformation to a delegate
    Delegate Transform(Delegate original);
    string Name { get; }
    TransformationType Type { get; }
}

public enum TransformationType
{
    Vectorization,      // vmap
    Parallelization,    // pmap
    Compilation,        // jit
    Composition         // compose/partial
}

// Context for transformation execution
public class TransformationContext
{
    public bool DebugMode { get; set; }
    public Dictionary<string, object> Metadata { get; }
    public TransformationContext Parent { get; }
}

// Function metadata attribute
[AttributeUsage(AttributeTargets.Method | AttributeTargets.Delegate)]
public class TensorFunctionAttribute : Attribute
{
    public bool IsPure { get; set; } = true;
    public string[] InputShapes { get; set; }
    public string OutputShape { get; set; }
}
```

### 2. Base Transformation Class

```csharp
public abstract class BaseTransformation : IFunctionalTransformation
{
    public string Name { get; }
    public TransformationType Type { get; }
    protected TransformationContext Context { get; }

    protected BaseTransformation(string name, TransformationType type, TransformationContext context = null)
    {
        Name = name;
        Type = type;
        Context = context ?? new TransformationContext();
    }

    public abstract Delegate Transform(Delegate original);

    protected void ValidateDelegate(Delegate del)
    {
        if (del == null)
            throw new ArgumentNullException(nameof(del));

        // Check if delegate has TensorFunction attribute or accepts Tensor parameters
        var method = del.Method;
        var parameters = method.GetParameters();

        if (!parameters.Any(p => typeof(Tensor).IsAssignableFrom(p.ParameterType)))
        {
            throw new ArgumentException("Function must accept at least one Tensor parameter");
        }
    }
}
```

### 3. Functional Utilities Namespace

```csharp
namespace MLFramework.Functional
{
    // Main entry point class (to be filled by other specs)
    public static class Functional
    {
        // Methods will be added in subsequent specs:
        // - Vectorize
        // - Parallelize
        // - Compile
        // - Compose
    }
}
```

### 4. Transformation Registry

```csharp
public class TransformationRegistry
{
    private static readonly ConcurrentDictionary<Delegate, List<IFunctionalTransformation>> _transformations =
        new ConcurrentDictionary<Delegate, List<IFunctionalTransformation>>();

    public static void Register(Delegate original, IFunctionalTransformation transform)
    {
        _transformations.AddOrUpdate(
            original,
            _ => new List<IFunctionalTransformation> { transform },
            (_, list) =>
            {
                list.Add(transform);
                return list;
            });
    }

    public static List<IFunctionalTransformation> GetTransformations(Delegate original)
    {
        _transformations.TryGetValue(original, out var transforms);
        return transforms ?? new List<IFunctionalTransformation>();
    }

    public static void Clear()
    {
        _transformations.Clear();
    }
}
```

## Files to Create
1. `src/MLFramework/Functional/IFunctionalTransformation.cs`
2. `src/MLFramework/Functional/BaseTransformation.cs`
3. `src/MLFramework/Functional/TransformationContext.cs`
4. `src/MLFramework/Functional/TensorFunctionAttribute.cs`
5. `src/MLFramework/Functional/TransformationRegistry.cs`
6. `src/MLFramework/Functional/Functional.cs` (placeholder, to be extended)

## Dependencies
- MLFramework.Tensor (basic Tensor type)
- System.Collections.Concurrent

## Success Criteria
- Interfaces compile without errors
- Base class provides validation logic
- Registry can track transformations
- Transformation metadata is properly structured

## Notes for Coder
- This is foundational infrastructure - keep it simple and extensible
- Focus on abstractions, not implementation details
- All concrete implementations will be in subsequent specs
- Include XML documentation for all public APIs
