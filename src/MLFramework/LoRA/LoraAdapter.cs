using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.LoRA;
using RitterFramework.Core.Tensor;

namespace MLFramework.LoRA;

/// <summary>
/// Represents a single LoRA adapter (fine-tuned task) with weights, configuration, and metadata.
/// </summary>
public class LoraAdapter
{
    /// <summary>Adapter name/identifier</summary>
    public string Name { get; set; }

    /// <summary>LoRA configuration</summary>
    public LoraConfig Config { get; set; }

    /// <summary>LoRA module weights keyed by module name</summary>
    public Dictionary<string, LoraModuleWeights> Weights { get; set; }

    /// <summary>Adapter metadata (training info, date, etc.)</summary>
    public AdapterMetadata Metadata { get; set; }

    /// <summary>
    /// Initializes a new instance of the LoraAdapter class.
    /// </summary>
    /// <param name="name">Adapter name/identifier</param>
    /// <param name="config">LoRA configuration</param>
    public LoraAdapter(string name, LoraConfig config)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Config = config ?? throw new ArgumentNullException(nameof(config));
        Weights = new Dictionary<string, LoraModuleWeights>();
        Metadata = new AdapterMetadata();
    }

    /// <summary>
    /// Add LoRA weights for a module.
    /// </summary>
    /// <param name="moduleName">Name of the module</param>
    /// <param name="loraA">LoRA matrix A tensor [out_features, rank]</param>
    /// <param name="loraB">LoRA matrix B tensor [rank, in_features]</param>
    public void AddModuleWeights(string moduleName, Tensor loraA, Tensor loraB)
    {
        if (string.IsNullOrEmpty(moduleName))
            throw new ArgumentException("Module name cannot be null or empty", nameof(moduleName));
        if (loraA == null)
            throw new ArgumentNullException(nameof(loraA));
        if (loraB == null)
            throw new ArgumentNullException(nameof(loraB));

        Weights[moduleName] = new LoraModuleWeights
        {
            LoraA = loraA.Clone(),
            LoraB = loraB.Clone()
        };

        // Update timestamp
        Metadata.UpdatedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Get LoRA weights for a module.
    /// </summary>
    /// <param name="moduleName">Name of the module</param>
    /// <param name="weights">Output weights if found</param>
    /// <returns>True if weights found, false otherwise</returns>
    public bool TryGetModuleWeights(string moduleName, out LoraModuleWeights weights)
    {
        if (string.IsNullOrEmpty(moduleName))
        {
            weights = null;
            return false;
        }

        return Weights.TryGetValue(moduleName, out weights);
    }

    /// <summary>
    /// Get total number of parameters in this adapter.
    /// </summary>
    /// <returns>Total parameter count</returns>
    public long GetParameterCount()
    {
        return Weights.Values.Sum(w => w.LoraA.Size + w.LoraB.Size);
    }

    /// <summary>
    /// Calculate memory size in bytes.
    /// </summary>
    /// <returns>Memory size in bytes</returns>
    public long GetMemorySize()
    {
        return GetParameterCount() * sizeof(float);
    }

    /// <summary>
    /// Create a copy of this adapter.
    /// </summary>
    /// <returns>A new LoraAdapter instance with copied weights and metadata</returns>
    public LoraAdapter Clone()
    {
        var clonedAdapter = new LoraAdapter(this.Name, this.Config.Clone())
        {
            Metadata = new AdapterMetadata
            {
                CreatedAt = this.Metadata.CreatedAt,
                UpdatedAt = this.Metadata.UpdatedAt,
                BaseModel = this.Metadata.BaseModel,
                TrainingEpochs = this.Metadata.TrainingEpochs,
                FinalLoss = this.Metadata.FinalLoss,
                CustomFields = new Dictionary<string, string>(this.Metadata.CustomFields)
            }
        };

        // Clone all weights
        foreach (var kvp in this.Weights)
        {
            clonedAdapter.Weights[kvp.Key] = new LoraModuleWeights
            {
                LoraA = kvp.Value.LoraA.Clone(),
                LoraB = kvp.Value.LoraB.Clone()
            };
        }

        return clonedAdapter;
    }
}

/// <summary>
/// Represents LoRA weights for a single module.
/// </summary>
public class LoraModuleWeights
{
    /// <summary>LoRA matrix A tensor [out_features, rank]</summary>
    public Tensor LoraA { get; set; }

    /// <summary>LoRA matrix B tensor [rank, in_features]</summary>
    public Tensor LoraB { get; set; }
}

/// <summary>
/// Metadata for a LoRA adapter.
/// </summary>
public class AdapterMetadata
{
    /// <summary>Creation timestamp</summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>Last update timestamp</summary>
    public DateTime? UpdatedAt { get; set; }

    /// <summary>Base model name/identifier</summary>
    public string BaseModel { get; set; }

    /// <summary>Number of training epochs (if trained)</summary>
    public int? TrainingEpochs { get; set; }

    /// <summary>Final loss value (if trained)</summary>
    public float? FinalLoss { get; set; }

    /// <summary>Custom metadata fields</summary>
    public Dictionary<string, string> CustomFields { get; set; } = new();

    /// <summary>
    /// Add or update a custom metadata field.
    /// </summary>
    /// <param name="key">Field key</param>
    /// <param name="value">Field value</param>
    public void SetCustomField(string key, string value)
    {
        if (string.IsNullOrEmpty(key))
            throw new ArgumentException("Key cannot be null or empty", nameof(key));

        CustomFields[key] = value ?? string.Empty;
    }

    /// <summary>
    /// Get a custom metadata field.
    /// </summary>
    /// <param name="key">Field key</param>
    /// <param name="value">Output value if found</param>
    /// <returns>True if field found, false otherwise</returns>
    public bool TryGetCustomField(string key, out string value)
    {
        if (string.IsNullOrEmpty(key))
        {
            value = null;
            return false;
        }

        return CustomFields.TryGetValue(key, out value);
    }
}
