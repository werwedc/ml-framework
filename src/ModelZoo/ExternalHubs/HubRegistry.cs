namespace MLFramework.ModelZoo.ExternalHubs;

/// <summary>
/// Registry for managing model hubs.
/// </summary>
public class HubRegistry
{
    private readonly Dictionary<string, IModelHub> _hubs;
    private IModelHub? _defaultHub;

    /// <summary>
    /// Initializes a new instance of the HubRegistry class.
    /// </summary>
    public HubRegistry()
    {
        _hubs = new Dictionary<string, IModelHub>(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Registers a model hub.
    /// </summary>
    /// <param name="hub">The hub to register.</param>
    /// <exception cref="ArgumentNullException">Thrown when hub is null.</exception>
    /// <exception cref="ArgumentException">Thrown when a hub with the same name is already registered.</exception>
    public void RegisterHub(IModelHub hub)
    {
        if (hub == null)
        {
            throw new ArgumentNullException(nameof(hub));
        }

        var hubName = hub.HubName;
        if (string.IsNullOrWhiteSpace(hubName))
        {
            throw new ArgumentException("Hub name cannot be null or empty.", nameof(hub));
        }

        if (_hubs.ContainsKey(hubName))
        {
            throw new ArgumentException($"A hub with the name '{hubName}' is already registered.");
        }

        _hubs[hubName] = hub;

        // Set as default if it's the first hub
        if (_defaultHub == null)
        {
            _defaultHub = hub;
        }
    }

    /// <summary>
    /// Unregisters a model hub by name.
    /// </summary>
    /// <param name="hubName">The name of the hub to unregister.</param>
    /// <returns>True if the hub was successfully unregistered; false if the hub was not found.</returns>
    public bool UnregisterHub(string hubName)
    {
        if (string.IsNullOrWhiteSpace(hubName))
        {
            return false;
        }

        if (!_hubs.Remove(hubName))
        {
            return false;
        }

        // Update default hub if needed
        if (_defaultHub != null && _defaultHub.HubName.Equals(hubName, StringComparison.OrdinalIgnoreCase))
        {
            _defaultHub = _hubs.Values.FirstOrDefault();
        }

        return true;
    }

    /// <summary>
    /// Gets a registered hub by name.
    /// </summary>
    /// <param name="hubName">The name of the hub to get.</param>
    /// <returns>The hub if found; otherwise, null.</returns>
    public IModelHub? GetHub(string hubName)
    {
        if (string.IsNullOrWhiteSpace(hubName))
        {
            return null;
        }

        _hubs.TryGetValue(hubName, out var hub);
        return hub;
    }

    /// <summary>
    /// Gets the appropriate hub for a given model ID.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>The hub that can handle the model ID; otherwise, the default hub.</returns>
    public IModelHub? GetHubForModel(string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            return _defaultHub;
        }

        // Try to find a hub that can handle this model ID
        foreach (var hub in _hubs.Values)
        {
            if (hub.CanHandle(modelId))
            {
                return hub;
            }
        }

        // Return default hub if no specific hub found
        return _defaultHub;
    }

    /// <summary>
    /// Lists all registered hub names.
    /// </summary>
    /// <returns>An array of registered hub names.</returns>
    public string[] ListHubs()
    {
        return _hubs.Keys.ToArray();
    }

    /// <summary>
    /// Gets all registered hubs.
    /// </summary>
    /// <returns>An array of all registered hubs.</returns>
    public IModelHub[] GetAllHubs()
    {
        return _hubs.Values.ToArray();
    }

    /// <summary>
    /// Gets the default hub.
    /// </summary>
    /// <returns>The default hub; otherwise, null.</returns>
    public IModelHub? GetDefaultHub()
    {
        return _defaultHub;
    }

    /// <summary>
    /// Sets the default hub.
    /// </summary>
    /// <param name="hubName">The name of the hub to set as default.</param>
    /// <returns>True if the default hub was successfully set; false if the hub was not found.</returns>
    public bool SetDefaultHub(string hubName)
    {
        if (string.IsNullOrWhiteSpace(hubName))
        {
            return false;
        }

        var hub = GetHub(hubName);
        if (hub == null)
        {
            return false;
        }

        _defaultHub = hub;
        return true;
    }

    /// <summary>
    /// Checks if a hub with the given name is registered.
    /// </summary>
    /// <param name="hubName">The name of the hub.</param>
    /// <returns>True if the hub is registered; otherwise, false.</returns>
    public bool IsHubRegistered(string hubName)
    {
        if (string.IsNullOrWhiteSpace(hubName))
        {
            return false;
        }

        return _hubs.ContainsKey(hubName);
    }

    /// <summary>
    /// Gets the number of registered hubs.
    /// </summary>
    public int Count => _hubs.Count;

    /// <summary>
    /// Clears all registered hubs.
    /// </summary>
    public void Clear()
    {
        _hubs.Clear();
        _defaultHub = null;
    }

    /// <summary>
    /// Registers the default built-in hubs.
    /// </summary>
    public void RegisterDefaultHubs()
    {
        RegisterHub(new HuggingFaceHub());
        RegisterHub(new TensorFlowHub());
        RegisterHub(new ONNXHub());
    }
}
