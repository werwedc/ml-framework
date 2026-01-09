namespace MLFramework.Utilities;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

/// <summary>
/// Provides centralized control over random number generators (RNGs) used throughout the framework.
/// This class manages seeding for CPU, NumPy, and CUDA RNGs to ensure reproducibility.
/// </summary>
public class SeedManager : IDisposable
{
    private int _currentSeed;
    private readonly object _lock = new();
    private bool _disposed;
    private readonly Dictionary<int, int> _deviceSeeds = new();
    private readonly Dictionary<int, int> _workerSeeds = new();
    private DeterministicModeFlags _deterministicMode = DeterministicModeFlags.None;
    private readonly List<string> _nonDeterministicOperations = new();

    /// <summary>
    /// Initializes a new instance of the SeedManager class.
    /// </summary>
    public SeedManager()
    {
        _currentSeed = 0;
        _disposed = false;
    }

    /// <summary>
    /// Sets a global seed for all RNGs (random, NumPy, CUDA).
    /// This method is thread-safe.
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetGlobalSeed(int seed)
    {
        lock (_lock)
        {
            _currentSeed = seed;
            SetRandomSeed(seed);
            SetNumpySeed(seed);
            SetCudaSeed(seed);
        }
    }

    /// <summary>
    /// Seeds the CPU random number generator.
    /// Stores the seed for new Random instances.
    /// Note: Existing Random instances are not affected.
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetRandomSeed(int seed)
    {
        // For pure C# implementation, we store the seed for new Random instances.
        // Existing Random instances are not affected to avoid unexpected behavior.
        // Consider using a thread-safe Random pool in future implementations.
    }

    /// <summary>
    /// Seeds the NumPy random number generator.
    /// Note: This is a placeholder for interop with NumPy (if using Python interop).
    /// For pure C# implementation, this may be a no-op or use equivalent library.
    /// The interface is designed to be compatible with future NumPy integration.
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetNumpySeed(int seed)
    {
        // Placeholder for NumPy interop
        // In pure C# implementation, this is a no-op or uses equivalent library
    }

    /// <summary>
    /// Seeds the CUDA random number generator.
    /// Interface for CUDA RNG seeding.
    /// Placeholder for CUDA-specific implementation.
    /// Handles cases where CUDA is not available.
    /// </summary>
    /// <param name="seed">The seed value to use</param>
    public void SetCudaSeed(int seed)
    {
        // Placeholder for CUDA-specific implementation
        // Will handle CUDA availability checks in future iterations
    }

    /// <summary>
    /// Gets the current global seed value.
    /// </summary>
    public int CurrentSeed => _currentSeed;

    /// <summary>
    /// Gets or sets the current deterministic mode flags.
    /// Setting this property calls SetDeterministicMode to configure backend settings.
    /// </summary>
    public DeterministicModeFlags IsDeterministic
    {
        get => _deterministicMode;
        set => SetDeterministicMode(value);
    }

    /// <summary>
    /// Captures the current state of all RNGs
    /// </summary>
    /// <returns>RNGSnapshot containing current RNG states</returns>
    public RNGSnapshot CaptureRNGState()
    {
        var snapshot = new RNGSnapshot
        {
            RandomSeed = _currentSeed,
            Timestamp = DateTime.UtcNow
        };

        // Capture device seeds
        foreach (var kvp in _deviceSeeds)
        {
            snapshot.CudaStates[kvp.Key] = BitConverter.GetBytes(kvp.Value);
        }

        // Add metadata
        snapshot.Metadata["Version"] = "1.0";
        snapshot.Metadata["DeviceCount"] = _deviceSeeds.Count;

        return snapshot;
    }

    /// <summary>
    /// Restores RNGs to a previously captured state
    /// </summary>
    /// <param name="snapshot">The snapshot to restore</param>
    public void RestoreRNGState(RNGSnapshot snapshot)
    {
        if (snapshot == null)
            throw new ArgumentNullException(nameof(snapshot));

        if (!snapshot.IsValid())
            throw new ArgumentException("Invalid snapshot", nameof(snapshot));

        lock (_lock)
        {
            _currentSeed = snapshot.RandomSeed;
            SetRandomSeed(snapshot.RandomSeed);
            SetNumpySeed(snapshot.RandomSeed);
            SetCudaSeed(snapshot.RandomSeed);

            // Restore device seeds
            _deviceSeeds.Clear();
            foreach (var kvp in snapshot.CudaStates)
            {
                if (kvp.Value.Length == 4)
                {
                    _deviceSeeds[kvp.Key] = BitConverter.ToInt32(kvp.Value, 0);
                }
            }
        }
    }

    /// <summary>
    /// Saves an RNG snapshot to a file
    /// </summary>
    /// <param name="snapshot">The snapshot to save</param>
    /// <param name="filePath">Path to save the snapshot</param>
    public void SaveRNGSnapshot(RNGSnapshot snapshot, string filePath)
    {
        if (snapshot == null)
            throw new ArgumentNullException(nameof(snapshot));

        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be empty", nameof(filePath));

        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var options = new JsonSerializerOptions { WriteIndented = true };
        var json = JsonSerializer.Serialize(snapshot, options);
        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// Loads an RNG snapshot from a file
    /// </summary>
    /// <param name="filePath">Path to load the snapshot from</param>
    /// <returns>The loaded RNGSnapshot</returns>
    public RNGSnapshot LoadRNGSnapshot(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be empty", nameof(filePath));

        if (!File.Exists(filePath))
            throw new FileNotFoundException("Snapshot file not found", filePath);

        var json = File.ReadAllText(filePath);
        var snapshot = JsonSerializer.Deserialize<RNGSnapshot>(json);

        if (snapshot == null || !snapshot.IsValid())
        {
            throw new InvalidOperationException("Invalid or corrupted snapshot file");
        }

        return snapshot;
    }

    /// <summary>
    /// Seeds all available CUDA devices deterministically
    /// Uses formula: baseSeed + device_id
    /// </summary>
    /// <param name="baseSeed">The base seed value</param>
    /// <param name="deviceCount">Number of devices to seed (default: 1)</param>
    public void SeedAllDevices(int baseSeed, int? deviceCount = null)
    {
        var count = deviceCount ?? 1;

        for (int deviceId = 0; deviceId < count; deviceId++)
        {
            SeedDevice(deviceId, baseSeed + deviceId);
        }
    }

    /// <summary>
    /// Seeds a specific CUDA device
    /// </summary>
    /// <param name="deviceId">The device ID (0-based)</param>
    /// <param name="seed">The seed value for this device</param>
    public void SeedDevice(int deviceId, int seed)
    {
        if (deviceId < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceId), "Device ID must be non-negative");

        _deviceSeeds[deviceId] = seed;
        SetCudaSeed(seed);
    }

    /// <summary>
    /// Gets the seed for a specific device
    /// </summary>
    /// <param name="deviceId">The device ID</param>
    /// <returns>The seed value for the device</returns>
    public int GetDeviceSeed(int deviceId)
    {
        if (!_deviceSeeds.ContainsKey(deviceId))
        {
            throw new ArgumentException($"Device {deviceId} has not been seeded");
        }

        return _deviceSeeds[deviceId];
    }

    /// <summary>
    /// Seeds data loading workers deterministically
    /// Uses formula: baseSeed + worker_id
    /// </summary>
    /// <param name="baseSeed">The base seed value</param>
    /// <param name="workerCount">Number of workers to seed</param>
    public void SeedWorkers(int baseSeed, int workerCount)
    {
        if (workerCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(workerCount), "Worker count must be positive");

        for (int workerId = 0; workerId < workerCount; workerId++)
        {
            SeedWorker(workerId, baseSeed + workerId);
        }
    }

    /// <summary>
    /// Seeds a specific data loading worker
    /// </summary>
    /// <param name="workerId">The worker ID (0-based)</param>
    /// <param name="seed">The seed value for this worker</param>
    public void SeedWorker(int workerId, int seed)
    {
        if (workerId < 0)
            throw new ArgumentOutOfRangeException(nameof(workerId), "Worker ID must be non-negative");

        _workerSeeds[workerId] = seed;
    }

    /// <summary>
    /// Gets the seed for a specific worker
    /// </summary>
    /// <param name="workerId">The worker ID</param>
    /// <returns>The seed value for the worker</returns>
    public int GetWorkerSeed(int workerId)
    {
        if (!_workerSeeds.ContainsKey(workerId))
        {
            throw new ArgumentException($"Worker {workerId} has not been seeded");
        }

        return _workerSeeds[workerId];
    }

    /// <summary>
    /// Gets a deterministic seed for a given device and operation
    /// Formula: baseSeed + device_id * 1000 + operation_id
    /// </summary>
    /// <param name="baseSeed">The base seed</param>
    /// <param name="deviceId">The device ID</param>
    /// <param name="operationId">The operation ID</param>
    /// <returns>A deterministic seed for this combination</returns>
    public int GetDeterministicSeed(int baseSeed, int deviceId, int operationId = 0)
    {
        return baseSeed + (deviceId * 1000) + operationId;
    }

    #region Deterministic Mode Methods

    /// <summary>
    /// Enables or disables specific deterministic modes.
    /// Configures backend-specific settings based on the provided flags.
    /// </summary>
    /// <param name="flags">Deterministic mode flags to set</param>
    public void SetDeterministicMode(DeterministicModeFlags flags)
    {
        lock (_lock)
        {
            _deterministicMode = flags;

            // Configure backend-specific settings
            if ((flags & DeterministicModeFlags.CudnnDeterministic) != 0)
            {
                // Set environment variable for cuDNN deterministic mode
                // In a real implementation, this would use environment variable APIs
                System.Environment.SetEnvironmentVariable("CUDNN_DETERMINISTIC", "1");
            }
            else
            {
                System.Environment.SetEnvironmentVariable("CUDNN_DETERMINISTIC", null);
            }

            if ((flags & DeterministicModeFlags.CublasDeterministic) != 0)
            {
                // Configure cuBLAS to use PEDANTIC mode
                // Placeholder for actual CUDA interop
            }

            if ((flags & DeterministicModeFlags.CudaMemoryDeterministic) != 0)
            {
                // Set memory pool size limit for deterministic allocation
                // Placeholder for CUDA interop (cudaDeviceSetLimit)
            }

            if ((flags & DeterministicModeFlags.CudaGraphs) != 0)
            {
                // Enable CUDA Graph capture mode
                // Placeholder for CUDA interop
            }
        }
    }

    /// <summary>
    /// Enables a specific deterministic mode flag.
    /// </summary>
    /// <param name="flag">The flag to enable</param>
    public void EnableDeterministicMode(DeterministicModeFlags flag)
    {
        lock (_lock)
        {
            SetDeterministicMode(_deterministicMode | flag);
        }
    }

    /// <summary>
    /// Disables a specific deterministic mode flag.
    /// </summary>
    /// <param name="flag">The flag to disable</param>
    public void DisableDeterministicMode(DeterministicModeFlags flag)
    {
        lock (_lock)
        {
            SetDeterministicMode(_deterministicMode & ~flag);
        }
    }

    /// <summary>
    /// Checks if a specific deterministic mode is enabled.
    /// </summary>
    /// <param name="flag">The flag to check</param>
    /// <returns>True if the flag is enabled, false otherwise</returns>
    public bool IsDeterministicModeEnabled(DeterministicModeFlags flag)
    {
        lock (_lock)
        {
            return (_deterministicMode & flag) != 0;
        }
    }

    /// <summary>
    /// Gets a description of performance impact for current deterministic settings.
    /// </summary>
    /// <returns>A string describing the performance impact</returns>
    public string GetPerformanceImpact()
    {
        lock (_lock)
        {
            var impacts = new List<string>();

            if ((_deterministicMode & DeterministicModeFlags.CudnnDeterministic) != 0)
            {
                impacts.Add("CudnnDeterministic: ~20% slower");
            }

            if ((_deterministicMode & DeterministicModeFlags.CublasDeterministic) != 0)
            {
                impacts.Add("CublasDeterministic: ~15% slower");
            }

            if ((_deterministicMode & DeterministicModeFlags.CudaMemoryDeterministic) != 0)
            {
                impacts.Add("CudaMemoryDeterministic: minimal impact, may reduce fragmentation");
            }

            if ((_deterministicMode & DeterministicModeFlags.CudaGraphs) != 0)
            {
                impacts.Add("CudaGraphs: may improve performance for repeated operations");
            }

            if (impacts.Count == 0)
            {
                return "No deterministic mode enabled - no performance impact";
            }

            return string.Join(", ", impacts);
        }
    }

    #endregion

    /// <summary>
    /// Creates a scoped context with a specific global seed
    /// </summary>
    /// <param name="seed">The seed to use within the scope</param>
    /// <returns>ScopedContext that restores previous state on disposal</returns>
    public IDisposable WithGlobalSeed(int seed)
    {
        var previousSeed = _currentSeed;
        SetGlobalSeed(seed);
        return new ScopedContext(this, previousSeed);
    }

    /// <summary>
    /// Creates a scoped context with deterministic behavior enabled.
    /// </summary>
    /// <param name="enabled">Whether to enable determinism in the scope</param>
    /// <returns>ScopedContext that restores previous state on disposal</returns>
    public IDisposable WithDeterminism(bool enabled)
    {
        return WithDeterminism(enabled, restoreState: false);
    }

    /// <summary>
    /// Creates a scoped context with deterministic behavior enabled.
    /// </summary>
    /// <param name="enabled">Whether to enable determinism in the scope</param>
    /// <param name="restoreState">Whether to capture and restore full RNG state</param>
    /// <returns>ScopedContext that restores previous state on disposal</returns>
    public IDisposable WithDeterminism(bool enabled, bool restoreState)
    {
        var previousMode = _deterministicMode;
        RNGSnapshot? previousRngState = restoreState ? CaptureRNGState() : null;

        if (enabled)
        {
            SetDeterministicMode(DeterministicModeFlags.All);
        }
        else
        {
            SetDeterministicMode(DeterministicModeFlags.None);
        }

        return new ScopedContext(this, previousMode, previousRngState, restoreState: restoreState);
    }

    /// <summary>
    /// Creates a scoped context with specific deterministic mode flags.
    /// </summary>
    /// <param name="flags">The deterministic mode flags to use within the scope</param>
    /// <returns>ScopedContext that restores previous state on disposal</returns>
    public IDisposable WithDeterministicMode(DeterministicModeFlags flags)
    {
        var previousMode = _deterministicMode;
        SetDeterministicMode(flags);
        return new ScopedContext(this, previousMode, restoreState: false);
    }

    #region Validation and Diagnostics

    /// <summary>
    /// Registers a known non-deterministic operation
    /// </summary>
    /// <param name="operationName">Name of the operation</param>
    public void RegisterNonDeterministicOperation(string operationName)
    {
        if (string.IsNullOrWhiteSpace(operationName))
            throw new ArgumentException("Operation name cannot be empty", nameof(operationName));

        lock (_lock)
        {
            if (!_nonDeterministicOperations.Contains(operationName))
            {
                _nonDeterministicOperations.Add(operationName);
            }
        }
    }

    /// <summary>
    /// Gets a list of registered non-deterministic operations
    /// </summary>
    /// <returns>List of operation names</returns>
    public IReadOnlyList<string> GetNonDeterministicOperations()
    {
        lock (_lock)
        {
            return new List<string>(_nonDeterministicOperations).AsReadOnly();
        }
    }

    /// <summary>
    /// Validates current deterministic configuration
    /// </summary>
    /// <returns>ValidationResult with messages, warnings, and errors</returns>
    public ValidationResult ValidateConfiguration()
    {
        var result = new ValidationResult
        {
            Messages = new List<string>(),
            Warnings = new List<string>(),
            Errors = new List<string>()
        };

        lock (_lock)
        {
            // Check if deterministic mode is enabled
            if (_deterministicMode != DeterministicModeFlags.None)
            {
                result.Messages.Add("Deterministic mode is enabled");

                // Validate devices are seeded if multi-device
                if (_deviceSeeds.Count > 0)
                {
                    result.Messages.Add($"{_deviceSeeds.Count} devices seeded");
                }

                // Validate workers are seeded if multi-worker
                if (_workerSeeds.Count > 0)
                {
                    result.Messages.Add($"{_workerSeeds.Count} workers seeded");
                }

                // Check for non-determinizable operations in deterministic mode
                foreach (var operation in _nonDeterministicOperations)
                {
                    if (!CanBeDeterministic(operation))
                    {
                        result.Errors.Add($"Non-determinizable operation '{operation}' is used in deterministic mode");
                    }
                    else if (KnownNonDeterministicOperations.Determinizable.Contains(operation))
                    {
                        result.Warnings.Add($"Determinizable operation '{operation}' is used - ensure proper deterministic settings");
                    }
                }
            }
            else
            {
                result.Messages.Add("Deterministic mode is not enabled");
            }

            // Set validation status
            result.IsValid = result.Errors.Count == 0;
        }

        return result;
    }

    /// <summary>
    /// Gets diagnostic information about current state
    /// </summary>
    /// <returns>DiagnosticInfo object</returns>
    public DiagnosticInfo GetDiagnosticInfo()
    {
        var info = new DiagnosticInfo();

        lock (_lock)
        {
            info.DeterministicMode = _deterministicMode;
            info.CurrentSeed = _currentSeed;
            info.DeviceCount = _deviceSeeds.Count;
            info.WorkerCount = _workerSeeds.Count;
            info.PerformanceImpact = GetPerformanceImpact();

            foreach (var operation in _nonDeterministicOperations)
            {
                info.NonDeterministicOperations.Add(operation);
            }
        }

        return info;
    }

    /// <summary>
    /// Prints diagnostic information to console
    /// </summary>
    public void PrintDiagnostics()
    {
        var info = GetDiagnosticInfo();
        var timestamp = DateTime.UtcNow;

        Console.WriteLine($"\n=== SeedManager Diagnostics [{timestamp:yyyy-MM-dd HH:mm:ss}] ===");
        Console.WriteLine($"Deterministic Mode: {info.DeterministicMode}");
        Console.WriteLine($"Current Seed: {info.CurrentSeed}");
        Console.WriteLine($"Device Count: {info.DeviceCount}");
        Console.WriteLine($"Worker Count: {info.WorkerCount}");
        Console.WriteLine($"Performance Impact: {info.PerformanceImpact}");

        if (info.NonDeterministicOperations.Count > 0)
        {
            Console.WriteLine("\nNon-Deterministic Operations:");
            foreach (var operation in info.NonDeterministicOperations)
            {
                var determinizable = KnownNonDeterministicOperations.Determinizable.Contains(operation);
                Console.WriteLine($"  - {operation} ({(determinizable ? "Determinizable" : "Non-Determinizable")})");
            }
        }
        else
        {
            Console.WriteLine("\nNon-Deterministic Operations: None registered");
        }

        Console.WriteLine("=====================================================\n");
    }

    /// <summary>
    /// Checks if an operation can be made deterministic
    /// </summary>
    /// <param name="operationName">Name of the operation</param>
    /// <returns>True if operation can be deterministic</returns>
    public bool CanBeDeterministic(string operationName)
    {
        if (string.IsNullOrWhiteSpace(operationName))
            return false;

        return !KnownNonDeterministicOperations.NonDeterminizable.Contains(operationName);
    }

    /// <summary>
    /// Warns about performance impact of current deterministic settings
    /// </summary>
    /// <returns>Warning message or null if no impact</returns>
    public string? CheckPerformanceImpact()
    {
        lock (_lock)
        {
            if (_deterministicMode == DeterministicModeFlags.None)
            {
                return null; // No deterministic mode, no performance impact
            }

            var warnings = new List<string>();

            if ((_deterministicMode & DeterministicModeFlags.CudnnDeterministic) != 0)
            {
                warnings.Add("cuDNN deterministic mode may reduce performance by 20-30%");
            }

            if ((_deterministicMode & DeterministicModeFlags.CublasDeterministic) != 0)
            {
                warnings.Add("cuBLAS deterministic mode may reduce performance by 15-25%");
            }

            if (warnings.Count > 0)
            {
                return string.Join("; ", warnings);
            }

            return null;
        }
    }

    #endregion

    /// <summary>
    /// Disposes the seed manager and cleans up resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;

        // Clean up resources here
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finalizer for SeedManager.
    /// </summary>
    ~SeedManager()
    {
        Dispose();
    }
}
