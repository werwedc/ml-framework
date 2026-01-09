namespace MLFramework.Utilities;

using System;
using System.Collections.Generic;
using System.Text.Json;

/// <summary>
/// Serializable snapshot of all RNG states in the system
/// </summary>
[Serializable]
public class RNGSnapshot
{
    /// <summary>
    /// Timestamp when snapshot was created
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// CPU random generator state
    /// </summary>
    public int RandomSeed { get; set; }

    /// <summary>
    /// NumPy RNG state (serialized)
    /// </summary>
    public byte[]? NumpyState { get; set; }

    /// <summary>
    /// CUDA RNG state per device
    /// </summary>
    public Dictionary<int, byte[]> CudaStates { get; set; }

    /// <summary>
    /// Additional metadata for debugging
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; }

    public RNGSnapshot()
    {
        Timestamp = DateTime.UtcNow;
        CudaStates = new Dictionary<int, byte[]>();
        Metadata = new Dictionary<string, object>();
    }

    /// <summary>
    /// Validates the snapshot data
    /// </summary>
    public bool IsValid()
    {
        return CudaStates != null &&
               Metadata != null &&
               Timestamp != default;
    }
}
