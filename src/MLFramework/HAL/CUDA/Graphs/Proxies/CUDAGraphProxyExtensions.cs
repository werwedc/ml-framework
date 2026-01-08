using MLFramework.HAL.CUDA.Graphs.Attributes;
using MLFramework.HAL.CUDA.Graphs.Proxies;

namespace MLFramework.HAL.CUDA.Graphs.Proxies;

/// <summary>
/// Extension methods for creating CUDA graph proxies
/// </summary>
public static class CUDAGraphProxyExtensions
{
    /// <summary>
    /// Wraps an instance with automatic graph capture
    /// </summary>
    /// <typeparam name="T">The type of object to wrap</typeparam>
    /// <param name="instance">The instance to wrap</param>
    /// <param name="stream">The CUDA stream to use for graph execution</param>
    /// <param name="configure">Optional configuration action for the proxy</param>
    /// <returns>A proxied instance with graph capture enabled</returns>
    public static T WithGraphCapture<T>(
        this T instance,
        CudaStream stream,
        Action<CUDAGraphProxy<T>> configure = null) where T : class
    {
        if (instance == null)
            throw new ArgumentNullException(nameof(instance));
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        var proxy = new CUDAGraphProxy<T>(instance, stream);
        configure?.Invoke(proxy);

        return CUDAGraphProxy<T>.Create(instance, stream, proxy.GraphManager);
    }

    /// <summary>
    /// Wraps an instance with automatic graph capture using a custom manager
    /// </summary>
    /// <typeparam name="T">The type of object to wrap</typeparam>
    /// <param name="instance">The instance to wrap</param>
    /// <param name="stream">The CUDA stream to use for graph execution</param>
    /// <param name="manager">The custom graph manager to use</param>
    /// <returns>A proxied instance with graph capture enabled</returns>
    public static T WithGraphCapture<T>(
        this T instance,
        CudaStream stream,
        CUDAGraphManager manager) where T : class
    {
        if (instance == null)
            throw new ArgumentNullException(nameof(instance));
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        if (manager == null)
            throw new ArgumentNullException(nameof(manager));

        return CUDAGraphProxy<T>.Create(instance, stream, manager);
    }

    /// <summary>
    /// Creates a new CUDAGraphProxy with automatic graph capture
    /// </summary>
    /// <typeparam name="T">The type of object to proxy</typeparam>
    /// <param name="instance">The instance to proxy</param>
    /// <param name="stream">The CUDA stream to use for graph execution</param>
    /// <returns>A new CUDAGraphProxy instance</returns>
    public static CUDAGraphProxy<T> CreateGraphProxy<T>(
        this T instance,
        CudaStream stream) where T : class
    {
        if (instance == null)
            throw new ArgumentNullException(nameof(instance));
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        return new CUDAGraphProxy<T>(instance, stream);
    }

    /// <summary>
    /// Creates a new CUDAGraphProxy with automatic graph capture and custom manager
    /// </summary>
    /// <typeparam name="T">The type of object to proxy</typeparam>
    /// <param name="instance">The instance to proxy</param>
    /// <param name="stream">The CUDA stream to use for graph execution</param>
    /// <param name="manager">The custom graph manager to use</param>
    /// <returns>A new CUDAGraphProxy instance</returns>
    public static CUDAGraphProxy<T> CreateGraphProxy<T>(
        this T instance,
        CudaStream stream,
        CUDAGraphManager manager) where T : class
    {
        if (instance == null)
            throw new ArgumentNullException(nameof(instance));
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        if (manager == null)
            throw new ArgumentNullException(nameof(manager));

        return new CUDAGraphProxy<T>(instance, stream, manager);
    }
}
