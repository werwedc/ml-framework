# Spec: CaptureGraph Attribute

## Overview
Implement a C# attribute that can be applied to methods to enable automatic CUDA graph capture. This provides a declarative approach to graph capture, making it easier for users to enable graph optimizations.

## Requirements

### 1. CaptureGraph Attribute
Define the attribute for marking methods for graph capture.

```csharp
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public class CaptureGraphAttribute : Attribute
{
    /// <summary>
    /// Gets or sets the name of the graph
    /// </summary>
    public string GraphName { get; set; }

    /// <summary>
    /// Gets or sets the number of warm-up iterations before capture
    /// </summary>
    public int WarmupIterations { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to enable weight updates
    /// </summary>
    public bool EnableWeightUpdates { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable automatic fallback
    /// </summary>
    public bool EnableFallback { get; set; } = true;

    public CaptureGraphAttribute()
    {
        GraphName = string.Empty;
    }

    public CaptureGraphAttribute(string graphName)
    {
        GraphName = graphName;
    }
}
```

### 2. CUDAGraphProxy Class
Create a proxy that wraps methods with graph capture logic.

```csharp
public class CUDAGraphProxy<T> where T : class
{
    private readonly T _target;
    private readonly CUDAGraphManager _graphManager;
    private readonly CUDAStream _stream;
    private readonly Dictionary<string, MethodInfo> _graphMethods;

    public CUDAGraphProxy(T target, CUDAStream stream, CUDAGraphManager graphManager = null)
    {
        _target = target;
        _stream = stream;
        _graphManager = graphManager ?? new CUDAGraphManager();
        _graphMethods = new Dictionary<string, MethodInfo>();

        // Discover methods with CaptureGraph attribute
        DiscoverGraphMethods();
    }

    /// <summary>
    /// Creates a proxied instance that captures graphs automatically
    /// </summary>
    public static T Create(T instance, CUDAStream stream, CUDAGraphManager graphManager = null)
    {
        var proxy = new CUDAGraphProxy<T>(instance, stream, graphManager);
        return proxy.CreateProxy();
    }

    private T CreateProxy()
    {
        // Create dynamic proxy using DispatchProxy or similar mechanism
        // For simplicity, we'll use a more manual approach
        return CreateManualProxy();
    }

    private T CreateManualProxy()
    {
        // This is a simplified approach - in production, use
        // Castle DynamicProxy, DispatchProxy, or similar
        throw new NotImplementedException(
            "Use a proxy library like Castle DynamicProxy for production");
    }

    private void DiscoverGraphMethods()
    {
        var methods = typeof(T).GetMethods(
            BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);

        foreach (var method in methods)
        {
            var attr = method.GetCustomAttribute<CaptureGraphAttribute>();
            if (attr != null)
            {
                var graphName = string.IsNullOrEmpty(attr.GraphName)
                    ? $"{typeof(T).Name}.{method.Name}"
                    : attr.GraphName;
                _graphMethods[graphName] = method;
            }
        }
    }

    /// <summary>
    /// Executes a method with graph capture
    /// </summary>
    public TReturn ExecuteWithGraph<TReturn>(string graphName, params object[] args)
    {
        var method = _graphMethods[graphName];
        var attr = method.GetCustomAttribute<CaptureGraphAttribute>();

        if (attr.EnableFallback && !_graphManager.IsCaptureComplete)
        {
            // Execute normally during warm-up
            return (TReturn)method.Invoke(_target, args);
        }
        else
        {
            // Capture or execute with graph
            return _graphManager.ExecuteGraphOrFallback(
                graphName,
                stream => method.Invoke(_target, args),
                _stream) as TReturn ?? default;
        }
    }
}
```

### 3. Extension Methods for Easy Proxy Creation
Provide convenient extension methods.

```csharp
public static class CUDAGraphProxyExtensions
{
    /// <summary>
    /// Wraps an instance with automatic graph capture
    /// </summary>
    public static T WithGraphCapture<T>(
        this T instance,
        CUDAStream stream,
        Action<CUDAGraphProxy<T>> configure = null) where T : class
    {
        var proxy = new CUDAGraphProxy<T>(instance, stream);
        configure?.Invoke(proxy);
        return CUDAGraphProxy<T>.Create(instance, stream, proxy._graphManager);
    }

    /// <summary>
    /// Wraps an instance with automatic graph capture using a custom manager
    /// </summary>
    public static T WithGraphCapture<T>(
        this T instance,
        CUDAStream stream,
        CUDAGraphManager manager) where T : class
    {
        return CUDAGraphProxy<T>.Create(instance, stream, manager);
    }
}
```

## Implementation Details

### File Structure
- **File**: `src/CUDA/Graphs/Attributes/CaptureGraphAttribute.cs`
- **File**: `src/CUDA/Graphs/Proxies/CUDAGraphProxy.cs`
- **File**: `src/CUDA/Graphs/Proxies/CUDAGraphProxyExtensions.cs`

### Dependencies
- CaptureGraphAttribute class
- CUDAGraphManager (from spec_cuda_graph_manager)
- CUDAStream class (existing)
- System for Attribute, MethodInfo, Reflection
- System.Collections.Generic for Dictionary

### Proxy Strategy Options
1. **Castle DynamicProxy**: Recommended for production
2. **DispatchProxy**: Built-in but limited
3. **Manual Implementation**: Simplest but requires manual method calls
4. **Source Generation**: Best performance but complex

For this spec, we'll use manual implementation as the baseline.

### Usage Example

```csharp
public class Model
{
    [CaptureGraph("ForwardPass")]
    public Tensor Forward(Tensor input)
    {
        // Model forward pass
        var x = conv1.Forward(input);
        x = activation.Forward(x);
        return x;
    }
}

// Usage
var model = new Model();
var stream = new CUDAStream();

// Option 1: Use proxy
var proxiedModel = model.WithGraphCapture(stream);
var output = proxiedModel.Forward(input);

// Option 2: Use manager directly
var manager = new CUDAGraphManager();
manager.ExecutePhaseGraph(GraphPhase.Forward, s => model.Forward(input), stream);
```

## Success Criteria
- Attribute can be applied to methods
- Attribute properties are accessible
- Proxy can create wrapped instances
- Graph capture works through proxy
- Fallback works correctly
- Extension methods work as expected

## Testing Requirements

### Unit Tests
- Test attribute creation and properties
- Test proxy discovery of graph methods
- Test graph execution through proxy
- Test fallback behavior
- Test extension methods
- Test multiple methods with different graphs

### Integration Tests
- Test attribute with actual model execution (requires GPU)
- Test with multiple iterations
- Test with weight updates
