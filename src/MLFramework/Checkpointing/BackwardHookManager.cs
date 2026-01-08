using System;
using System.Collections.Generic;
using System.Linq;
using RitterTensor = RitterFramework.Core.Tensor;

namespace MLFramework.Checkpointing;

/// <summary>
/// Manages backward hooks for checkpointing
/// </summary>
public class BackwardHookManager : IDisposable
{
    private readonly Dictionary<int, (string LayerId, Action<RitterTensor.Tensor> Hook)> _hooks;
    private readonly Dictionary<string, List<int>> _layerToHandles;
    private int _nextHandle;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of BackwardHookManager
    /// </summary>
    public BackwardHookManager()
    {
        _hooks = new Dictionary<int, (string, Action<RitterTensor.Tensor>)>();
        _layerToHandles = new Dictionary<string, List<int>>();
        _nextHandle = 0;
        _disposed = false;
    }

    /// <summary>
    /// Registers a backward hook for a specific layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="hook">Hook function to call during backward pass</param>
    /// <returns>Handle that can be used to remove the hook</returns>
    public int RegisterHook(string layerId, Action<RitterTensor.Tensor> hook)
    {
        if (string.IsNullOrEmpty(layerId))
            throw new ArgumentException("Layer ID cannot be null or empty", nameof(layerId));
        if (hook == null)
            throw new ArgumentNullException(nameof(hook));
        ThrowIfDisposed();

        var handle = _nextHandle++;

        lock (_hooks)
        {
            _hooks[handle] = (layerId, hook);

            if (!_layerToHandles.ContainsKey(layerId))
            {
                _layerToHandles[layerId] = new List<int>();
            }
            _layerToHandles[layerId].Add(handle);
        }

        return handle;
    }

    /// <summary>
    /// Removes a previously registered hook
    /// </summary>
    /// <param name="handle">Handle returned from RegisterHook</param>
    public void RemoveHook(int handle)
    {
        ThrowIfDisposed();

        lock (_hooks)
        {
            if (!_hooks.TryGetValue(handle, out var hookInfo))
            {
                throw new ArgumentException($"Hook with handle {handle} not found", nameof(handle));
            }

            _hooks.Remove(handle);

            var layerId = hookInfo.LayerId;
            if (_layerToHandles.ContainsKey(layerId))
            {
                _layerToHandles[layerId].Remove(handle);
                if (_layerToHandles[layerId].Count == 0)
                {
                    _layerToHandles.Remove(layerId);
                }
            }
        }
    }

    /// <summary>
    /// Removes all hooks for a specific layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    public void RemoveHooksForLayer(string layerId)
    {
        if (string.IsNullOrEmpty(layerId))
            throw new ArgumentException("Layer ID cannot be null or empty", nameof(layerId));
        ThrowIfDisposed();

        lock (_hooks)
        {
            if (!_layerToHandles.TryGetValue(layerId, out var handles))
            {
                return;
            }

            foreach (var handle in handles)
            {
                _hooks.Remove(handle);
            }

            _layerToHandles.Remove(layerId);
        }
    }

    /// <summary>
    /// Removes all registered hooks
    /// </summary>
    public void ClearAllHooks()
    {
        ThrowIfDisposed();

        lock (_hooks)
        {
            _hooks.Clear();
            _layerToHandles.Clear();
        }
    }

    /// <summary>
    /// Invokes hooks for a specific layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="gradient">Gradient tensor</param>
    public void InvokeHooks(string layerId, RitterTensor.Tensor gradient)
    {
        if (string.IsNullOrEmpty(layerId))
            throw new ArgumentException("Layer ID cannot be null or empty", nameof(layerId));
        if (gradient == null)
            throw new ArgumentNullException(nameof(gradient));
        ThrowIfDisposed();

        List<Action<RitterTensor.Tensor>> hooksToInvoke;

        lock (_hooks)
        {
            if (!_layerToHandles.TryGetValue(layerId, out var handles))
            {
                return;
            }

            hooksToInvoke = handles
                .Where(handle => _hooks.ContainsKey(handle))
                .Select(handle => _hooks[handle].Hook)
                .ToList();
        }

        // Invoke hooks outside the lock to avoid deadlocks
        foreach (var hook in hooksToInvoke)
        {
            try
            {
                hook(gradient);
            }
            catch (Exception ex)
            {
                // Log the exception but continue invoking other hooks
                Console.Error.WriteLine($"Error invoking backward hook for layer {layerId}: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Gets the number of registered hooks
    /// </summary>
    public int HookCount
    {
        get
        {
            ThrowIfDisposed();
            lock (_hooks)
            {
                return _hooks.Count;
            }
        }
    }

    /// <summary>
    /// Gets the number of hooks for a specific layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>Number of hooks registered for the layer</returns>
    public int GetHookCountForLayer(string layerId)
    {
        if (string.IsNullOrEmpty(layerId))
            throw new ArgumentException("Layer ID cannot be null or empty", nameof(layerId));
        ThrowIfDisposed();

        lock (_hooks)
        {
            return _layerToHandles.TryGetValue(layerId, out var handles) ? handles.Count : 0;
        }
    }

    /// <summary>
    /// Checks if a hook is registered for the given handle
    /// </summary>
    /// <param name="handle">Handle to check</param>
    /// <returns>True if hook is registered, false otherwise</returns>
    public bool HasHook(int handle)
    {
        ThrowIfDisposed();

        lock (_hooks)
        {
            return _hooks.ContainsKey(handle);
        }
    }

    /// <summary>
    /// Disposes the manager and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            ClearAllHooks();
            _disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(BackwardHookManager));
        }
    }
}
