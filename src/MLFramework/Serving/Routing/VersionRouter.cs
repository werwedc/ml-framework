using Microsoft.Extensions.Logging;
using MLFramework.Serving.Deployment;

namespace MLFramework.Serving.Routing;

/// <summary>
/// Implementation of version router for directing inference requests to specific model versions
/// </summary>
public class VersionRouter : IVersionRouter
{
    private readonly IModelRegistry _modelRegistry;
    private readonly IModelLoader _modelLoader;
    private readonly ILogger<VersionRouter>? _logger;
    private readonly Dictionary<string, string> _defaultVersions;
    private readonly object _lock = new();

    public VersionRouter(
        IModelRegistry modelRegistry,
        IModelLoader modelLoader,
        ILogger<VersionRouter>? logger = null)
    {
        _modelRegistry = modelRegistry ?? throw new ArgumentNullException(nameof(modelRegistry));
        _modelLoader = modelLoader ?? throw new ArgumentNullException(nameof(modelLoader));
        _logger = logger;
        _defaultVersions = new Dictionary<string, string>();
    }

    /// <summary>
    /// Get a model based on routing context
    /// </summary>
    public IModel GetModel(string modelName, RoutingContext context)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));

        _logger?.LogDebug("Routing request for model: {ModelName}, context: {@Context}", modelName, context);

        // Check if context specifies a preferred version
        if (!string.IsNullOrWhiteSpace(context?.PreferredVersion))
        {
            _logger?.LogDebug("Using preferred version: {Version}", context.PreferredVersion);
            return GetModel(modelName, context.PreferredVersion);
        }

        // Try to use default version
        string? defaultVersion = GetDefaultVersion(modelName);
        if (!string.IsNullOrEmpty(defaultVersion))
        {
            _logger?.LogDebug("Using default version: {Version}", defaultVersion);
            return GetModel(modelName, defaultVersion);
        }

        // Fall back to latest version
        _logger?.LogDebug("No default version, using latest version");
        return GetLatestModel(modelName);
    }

    /// <summary>
    /// Get a model by specific version
    /// </summary>
    public IModel GetModel(string modelName, string version)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));

        if (string.IsNullOrWhiteSpace(version))
            throw new ArgumentException("Version cannot be null or empty", nameof(version));

        _logger?.LogDebug("Getting model: {ModelName}, version: {Version}", modelName, version);

        // Validate version exists in registry
        if (!_modelRegistry.HasVersion(modelName, version))
        {
            _logger?.LogError("Version {Version} not found for model {ModelName}", version, modelName);
            throw new RoutingException(modelName, version,
                $"Version '{version}' not found for model '{modelName}'");
        }

        // Try to get the loaded model
        var loadedModel = _modelLoader.GetLoadedModels()
            .FirstOrDefault(m => m.Name == modelName && m.Version == version);

        if (loadedModel == null)
        {
            _logger?.LogError("Model {ModelName} v{Version} is not loaded", modelName, version);
            throw new RoutingException(modelName, version,
                $"Model '{modelName}' version '{version}' is registered but not loaded");
        }

        _logger?.LogDebug("Successfully routed to model: {ModelName} v{Version}", modelName, version);
        return loadedModel;
    }

    /// <summary>
    /// Set the default version for a model
    /// </summary>
    public void SetDefaultVersion(string modelName, string version)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));

        if (string.IsNullOrWhiteSpace(version))
            throw new ArgumentException("Version cannot be null or empty", nameof(version));

        // Validate version exists
        if (!_modelRegistry.HasVersion(modelName, version))
        {
            throw new KeyNotFoundException(
                $"Version '{version}' not found for model '{modelName}'. Cannot set as default.");
        }

        lock (_lock)
        {
            _defaultVersions[modelName] = version;
            _logger?.LogInformation("Set default version for {ModelName} to {Version}", modelName, version);
        }
    }

    /// <summary>
    /// Get the default version for a model
    /// </summary>
    public string? GetDefaultVersion(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));

        lock (_lock)
        {
            _defaultVersions.TryGetValue(modelName, out var version);
            return version;
        }
    }

    /// <summary>
    /// Get the latest version of a model (highest semantic version)
    /// </summary>
    private IModel GetLatestModel(string modelName)
    {
        var versions = _modelRegistry.GetVersions(modelName).ToList();

        if (!versions.Any())
        {
            _logger?.LogError("No versions found for model: {ModelName}", modelName);
            throw new RoutingException(modelName, null,
                $"No versions found for model '{modelName}'");
        }

        // Find the latest version using semantic version comparison
        SemanticVersion? latestVersion = null;
        string? latestVersionStr = null;

        foreach (var versionStr in versions)
        {
            try
            {
                var version = SemanticVersion.Parse(versionStr);

                if (latestVersion == null || version > latestVersion.Value)
                {
                    latestVersion = version;
                    latestVersionStr = versionStr;
                }
            }
            catch (FormatException ex)
            {
                _logger?.LogWarning("Could not parse version '{Version}' as semantic version: {Error}",
                    versionStr, ex.Message);
                // Use the version as-is if it doesn't parse
                if (latestVersionStr == null)
                {
                    latestVersionStr = versionStr;
                }
            }
        }

        if (latestVersionStr == null)
        {
            _logger?.LogError("No valid versions found for model: {ModelName}", modelName);
            throw new RoutingException(modelName, null,
                $"No valid versions found for model '{modelName}'");
        }

        _logger?.LogDebug("Latest version for {ModelName} is {Version}", modelName, latestVersionStr);
        return GetModel(modelName, latestVersionStr);
    }
}
