namespace MLFramework.Serving.Deployment;

/// <summary>
/// Abstract base class providing common functionality for model implementations
/// </summary>
public abstract class BaseModel : IModel
{
    /// <inheritdoc/>
    public string Name { get; protected set; }

    /// <inheritdoc/>
    public string Version { get; protected set; }

    /// <inheritdoc/>
    public DateTime LoadTime { get; protected set; }

    /// <inheritdoc/>
    public bool IsActive { get; set; } = true;

    private bool _disposed;

    protected BaseModel(string name, string version)
    {
        Name = name;
        Version = version;
        LoadTime = DateTime.UtcNow;
    }

    /// <inheritdoc/>
    public abstract Task<InferenceResult> InferAsync(InferenceInput input);

    /// <summary>
    /// Dispose of the model's resources
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Dispose pattern implementation
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                // Dispose managed resources
                DisposeManagedResources();
            }

            // Dispose unmanaged resources
            DisposeUnmanagedResources();

            _disposed = true;
        }
    }

    /// <summary>
    /// Dispose of managed resources (override in derived classes if needed)
    /// </summary>
    protected virtual void DisposeManagedResources()
    {
    }

    /// <summary>
    /// Dispose of unmanaged resources (override in derived classes if needed)
    /// </summary>
    protected virtual void DisposeUnmanagedResources()
    {
    }

    /// <summary>
    /// Finalizer
    /// </summary>
    ~BaseModel()
    {
        Dispose(false);
    }
}
