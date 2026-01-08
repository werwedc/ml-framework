using System.Collections.Concurrent;

namespace MLFramework.Serving.Deployment;

/// <summary>
/// Implementation of model loading and unloading functionality
/// </summary>
public class ModelLoader : IModelLoader
{
    private readonly ConcurrentDictionary<string, IModel> _loadedModels;
    private readonly Func<string, string, IModel> _modelFactory;

    /// <summary>
    /// Create a new ModelLoader instance
    /// </summary>
    /// <param name="modelFactory">Factory function to create model instances (optional, for testing)</param>
    public ModelLoader(Func<string, string, IModel>? modelFactory = null)
    {
        _loadedModels = new ConcurrentDictionary<string, IModel>();
        _modelFactory = modelFactory ?? DefaultModelFactory;
    }

    /// <inheritdoc/>
    public IModel Load(string modelPath, string version)
    {
        // Validate model path exists
        if (!File.Exists(modelPath) && !Directory.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model path not found: {modelPath}", modelPath);
        }

        // Extract model name from path (use directory name as model name)
        var modelName = ExtractModelName(modelPath);

        // Check if model is already loaded
        var modelKey = GetModelKey(modelName, version);
        if (_loadedModels.ContainsKey(modelKey))
        {
            throw new InvalidOperationException(
                $"Model '{modelName}' version '{version}' is already loaded");
        }

        // Validate version format
        ValidateVersionFormat(version);

        // Create and track the model
        var model = _modelFactory(modelPath, version);
        if (!_loadedModels.TryAdd(modelKey, model))
        {
            throw new InvalidOperationException(
                $"Failed to load model '{modelName}' version '{version}' - concurrent load detected");
        }

        return model;
    }

    /// <inheritdoc/>
    public async Task<IModel> LoadAsync(string modelPath, string version, CancellationToken ct = default)
    {
        // Validate model path exists
        if (!File.Exists(modelPath) && !Directory.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model path not found: {modelPath}", modelPath);
        }

        // Extract model name from path
        var modelName = ExtractModelName(modelPath);

        // Check if model is already loaded
        var modelKey = GetModelKey(modelName, version);
        if (_loadedModels.ContainsKey(modelKey))
        {
            throw new InvalidOperationException(
                $"Model '{modelName}' version '{version}' is already loaded");
        }

        // Validate version format
        ValidateVersionFormat(version);

        // Check for cancellation
        ct.ThrowIfCancellationRequested();

        // Create and track the model
        var model = _modelFactory(modelPath, version);
        if (!_loadedModels.TryAdd(modelKey, model))
        {
            throw new InvalidOperationException(
                $"Failed to load model '{modelName}' version '{version}' - concurrent load detected");
        }

        // Simulate async load overhead (in real implementation, this would actually load async)
        await Task.Delay(1, ct);

        return model;
    }

    /// <inheritdoc/>
    public void Unload(IModel model)
    {
        var modelKey = GetModelKey(model.Name, model.Version);

        if (_loadedModels.TryRemove(modelKey, out var removedModel))
        {
            model.Dispose();
        }
        else
        {
            throw new InvalidOperationException(
                $"Model '{model.Name}' version '{model.Version}' is not loaded");
        }
    }

    /// <inheritdoc/>
    public bool IsLoaded(string name, string version)
    {
        var modelKey = GetModelKey(name, version);
        return _loadedModels.ContainsKey(modelKey);
    }

    /// <inheritdoc/>
    public IEnumerable<IModel> GetLoadedModels()
    {
        return _loadedModels.Values.ToList();
    }

    /// <summary>
    /// Default factory function for creating model instances
    /// </summary>
    private IModel DefaultModelFactory(string modelPath, string version)
    {
        // This is a placeholder - in a real implementation, this would:
        // 1. Detect model type from path/file extension
        // 2. Use appropriate model loader (ONNX, TensorFlow, PyTorch, etc.)
        // 3. Create actual model instance with loaded weights
        // For now, we create a mock model
        var modelName = ExtractModelName(modelPath);
        return new MockModel(modelName, version);
    }

    /// <summary>
    /// Extract model name from path
    /// </summary>
    private static string ExtractModelName(string modelPath)
    {
        var fileInfo = new FileInfo(modelPath);
        var dirInfo = new DirectoryInfo(modelPath);

        // If path is a file, use directory name as model name
        // If path is a directory, use directory name as model name
        return fileInfo.Exists ? fileInfo.Directory?.Name ?? fileInfo.Name : dirInfo.Name;
    }

    /// <summary>
    /// Get unique key for model name + version combination
    /// </summary>
    private static string GetModelKey(string name, string version)
    {
        return $"{name}:{version}";
    }

    /// <summary>
    /// Validate version format (basic semantic version check)
    /// </summary>
    private static void ValidateVersionFormat(string version)
    {
        if (string.IsNullOrWhiteSpace(version))
        {
            throw new ArgumentException("Version cannot be empty or whitespace", nameof(version));
        }

        // Basic check for semantic version format (major.minor.patch)
        if (!System.Text.RegularExpressions.Regex.IsMatch(version, @"^\d+\.\d+\.\d+.*$"))
        {
            throw new ArgumentException(
                $"Invalid version format '{version}'. Expected semantic version format (e.g., '1.0.0')",
                nameof(version));
        }
    }

    /// <summary>
    /// Mock model implementation for testing and placeholder purposes
    /// </summary>
    private class MockModel : BaseModel
    {
        public MockModel(string name, string version) : base(name, version)
        {
        }

        public override async Task<InferenceResult> InferAsync(InferenceInput input)
        {
            // Simulate inference latency
            await Task.Delay(1);

            // Return mock result
            return new InferenceResult(new { Input = input.Data, Timestamp = DateTime.UtcNow })
            {
                Success = true,
                InferenceTimeMs = 1
            };
        }
    }
}
