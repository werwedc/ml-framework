using System.Reflection;
using MLFramework.HAL.CUDA.Graphs.Attributes;

namespace MLFramework.HAL.CUDA.Graphs.Proxies;

/// <summary>
/// Proxy that wraps methods with graph capture logic
/// </summary>
/// <typeparam name="T">The type of object to proxy</typeparam>
public class CUDAGraphProxy<T> where T : class
{
    private readonly T _target;
    private readonly CudaStream _stream;
    private readonly CUDAGraphManager _graphManager;
    private readonly Dictionary<string, MethodInfo> _graphMethods;
    private readonly Dictionary<string, CaptureGraphAttribute> _graphAttributes;

    /// <summary>
    /// Gets the graph manager used by this proxy
    /// </summary>
    public CUDAGraphManager GraphManager => _graphManager;

    /// <summary>
    /// Creates a new CUDAGraphProxy
    /// </summary>
    /// <param name="target">The target object to proxy</param>
    /// <param name="stream">The CUDA stream to use for graph execution</param>
    /// <param name="graphManager">Optional custom graph manager</param>
    public CUDAGraphProxy(T target, CudaStream stream, CUDAGraphManager graphManager = null)
    {
        _target = target ?? throw new ArgumentNullException(nameof(target));
        _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        _graphManager = graphManager ?? new CUDAGraphManager();
        _graphMethods = new Dictionary<string, MethodInfo>();
        _graphAttributes = new Dictionary<string, CaptureGraphAttribute>();

        // Discover methods with CaptureGraph attribute
        DiscoverGraphMethods();
    }

    /// <summary>
    /// Creates a proxied instance that captures graphs automatically
    /// Note: This is a placeholder that returns the target. A full implementation
    /// would use a proxy library like Castle DynamicProxy or DispatchProxy
    /// </summary>
    public static T Create(T instance, CudaStream stream, CUDAGraphManager graphManager = null)
    {
        if (instance == null)
            throw new ArgumentNullException(nameof(instance));

        var proxy = new CUDAGraphProxy<T>(instance, stream, graphManager);
        // In production, use Castle DynamicProxy or DispatchProxy here
        return instance;
    }

    /// <summary>
    /// Discovers methods marked with CaptureGraph attribute
    /// </summary>
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
                _graphAttributes[graphName] = attr;
            }
        }
    }

    /// <summary>
    /// Gets the graph methods discovered by this proxy
    /// </summary>
    public IReadOnlyDictionary<string, MethodInfo> GetGraphMethods()
    {
        return _graphMethods;
    }

    /// <summary>
    /// Gets the attributes for discovered graph methods
    /// </summary>
    public IReadOnlyDictionary<string, CaptureGraphAttribute> GetGraphAttributes()
    {
        return _graphAttributes;
    }

    /// <summary>
    /// Executes a method with graph capture
    /// </summary>
    /// <typeparam name="TReturn">The return type of the method</typeparam>
    /// <param name="graphName">The name of the graph to execute</param>
    /// <param name="args">Arguments to pass to the method</param>
    /// <returns>The result of the method execution</returns>
    public TReturn ExecuteWithGraph<TReturn>(string graphName, params object[] args)
    {
        if (!_graphMethods.ContainsKey(graphName))
        {
            throw new KeyNotFoundException($"Graph '{graphName}' not found. " +
                $"Available graphs: {string.Join(", ", _graphMethods.Keys)}");
        }

        var method = _graphMethods[graphName];
        var attr = _graphAttributes[graphName];

        if (attr.EnableFallback && !_graphManager.IsCaptureComplete)
        {
            // Execute normally during warm-up
            return (TReturn)method.Invoke(_target, args);
        }
        else
        {
            // Capture or execute with graph
            var result = _graphManager.ExecuteGraphOrFallback(
                graphName,
                stream => method.Invoke(_target, args),
                _stream);

            return result is TReturn typedResult ? typedResult : default;
        }
    }

    /// <summary>
    /// Executes a method with graph capture (void return)
    /// </summary>
    /// <param name="graphName">The name of the graph to execute</param>
    /// <param name="args">Arguments to pass to the method</param>
    public void ExecuteWithGraph(string graphName, params object[] args)
    {
        if (!_graphMethods.ContainsKey(graphName))
        {
            throw new KeyNotFoundException($"Graph '{graphName}' not found. " +
                $"Available graphs: {string.Join(", ", _graphMethods.Keys)}");
        }

        var method = _graphMethods[graphName];
        var attr = _graphAttributes[graphName];

        if (attr.EnableFallback && !_graphManager.IsCaptureComplete)
        {
            // Execute normally during warm-up
            method.Invoke(_target, args);
        }
        else
        {
            // Capture or execute with graph
            _graphManager.ExecuteGraphOrFallback(
                graphName,
                stream => method.Invoke(_target, args),
                _stream);
        }
    }
}
