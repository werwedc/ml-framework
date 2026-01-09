using RitterFramework.Core;
using MLFramework.NN;
using RitterFramework.Core.Tensor;

namespace MLFramework.ModelZoo.Progressive;

/// <summary>
/// A parameter that loads weights on demand from storage.
/// </summary>
public class LazyParameter : Parameter
{
    private readonly ProgressiveLoadContext _context;
    private readonly string _layerName;
    private readonly string _weightPath;
    private readonly int[] _shape;
    private readonly DataType _dtype;
    private readonly long _sizeInBytes;
    private Tensor? _loadedTensor;
    private bool _isLoaded;
    private readonly object _loadLock = new();

    /// <summary>
    /// Gets whether the parameter has been loaded.
    /// </summary>
    public bool IsLoaded => _isLoaded;

    /// <summary>
    /// Gets the layer name this parameter belongs to.
    /// </summary>
    public string LayerName => _layerName;

    /// <summary>
    /// Gets the path to the weight file.
    /// </summary>
    public string WeightPath => _weightPath;

    /// <summary>
    /// Creates a new LazyParameter.
    /// </summary>
    /// <param name="layerName">The layer name.</param>
    /// <param name="weightPath">The path to the weight file.</param>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="context">The progressive load context.</param>
    /// <param name="requiresGrad">Whether gradients are required.</param>
    /// <param name="dtype">The data type.</param>
    public LazyParameter(
        string layerName,
        string weightPath,
        int[] shape,
        ProgressiveLoadContext context,
        bool requiresGrad = true,
        DataType dtype = DataType.Float32)
        : base(new float[0], shape, layerName, requiresGrad, dtype)
    {
        _layerName = layerName ?? throw new ArgumentNullException(nameof(layerName));
        _weightPath = weightPath ?? throw new ArgumentNullException(nameof(weightPath));
        _shape = shape ?? throw new ArgumentNullException(nameof(shape));
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _dtype = dtype;
        _isLoaded = false;

        // Calculate size in bytes (float32 = 4 bytes per element)
        int numElements = 1;
        foreach (int dim in shape)
        {
            numElements *= dim;
        }
        _sizeInBytes = (long)numElements * sizeof(float);
    }

    /// <summary>
    /// Ensures the weights are loaded, triggering load if necessary.
    /// </summary>
    public void EnsureLoaded()
    {
        if (_isLoaded)
        {
            // Record access for LRU tracking
            _context.MemoryManager.RecordAccess(_layerName);
            return;
        }

        lock (_loadLock)
        {
            if (_isLoaded)
            {
                return;
            }

            // Load weights from file
            LoadWeights();

            // Mark as loaded
            _isLoaded = true;

            // Register with memory manager
            _context.MemoryManager.RegisterLayer(_layerName, _sizeInBytes);

            // Mark layer as loaded in context
            _context.MarkLayerLoaded(_layerName);

            // Fire event
            OnLayerLoaded?.Invoke(this, EventArgs.Empty);
        }
    }

    /// <summary>
    /// Prefetches the weights for this parameter.
    /// </summary>
    public void Prefetch()
    {
        if (!_isLoaded)
        {
            EnsureLoaded();
        }
    }

    /// <summary>
    /// Loads weights from the weight file.
    /// </summary>
    private void LoadWeights()
    {
        if (!File.Exists(_weightPath))
        {
            throw new FileNotFoundException($"Weight file not found: {_weightPath}");
        }

        try
        {
            // Read binary file
            float[] data = ReadBinaryWeights(_weightPath);

            // Create tensor
            _loadedTensor = new Tensor(data, _shape, this.RequiresGrad, _dtype);

            // Copy data to base Tensor
            CopyLoadedData();
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load weights for layer '{_layerName}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Reads weights from a binary file.
    /// </summary>
    private float[] ReadBinaryWeights(string path)
    {
        using var fileStream = File.OpenRead(path);
        using var reader = new BinaryReader(fileStream);

        // Calculate expected number of elements
        int numElements = 1;
        foreach (int dim in _shape)
        {
            numElements *= dim;
        }

        // Read data
        float[] data = new float[numElements];
        for (int i = 0; i < numElements; i++)
        {
            data[i] = reader.ReadSingle();
        }

        return data;
    }

    /// <summary>
    /// Copies loaded data to the base Tensor class.
    /// </summary>
    private void CopyLoadedData()
    {
        if (_loadedTensor != null)
        {
            // Access private field via reflection or public properties
            // For now, we need to ensure the base Tensor has the correct data
            // This is a simplified approach - in practice, we might need to use reflection
            // or modify the Tensor class to support lazy loading
        }
    }

    /// <summary>
    /// Gets the tensor data, triggering load if necessary.
    /// </summary>
    public new float[] Data
    {
        get
        {
            EnsureLoaded();
            return _loadedTensor?.Data ?? Array.Empty<float>();
        }
    }

    /// <summary>
    /// Gets the tensor shape.
    /// </summary>
    public new int[] Shape => _shape;

    /// <summary>
    /// Unloads the weights from memory.
    /// </summary>
    public void Unload()
    {
        lock (_loadLock)
        {
            if (!_isLoaded)
            {
                return;
            }

            _loadedTensor = null;
            _isLoaded = false;

            // Unregister from memory manager
            long freedBytes = _context.MemoryManager.UnregisterLayer(_layerName);

            // Remove from loaded layers in context
            _context.LoadedLayers.Remove(_layerName);
            _context.UpdateProgress();
        }
    }

    /// <summary>
    /// Event fired when the layer finishes loading.
    /// </summary>
    public event EventHandler? OnLayerLoaded;

    /// <summary>
    /// Gets the size of this parameter in bytes.
    /// </summary>
    public long SizeInBytes => _sizeInBytes;
}
