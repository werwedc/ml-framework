using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.QAT;

/// <summary>
/// Manages layer-wise quantization configuration.
/// Allows overriding global configuration for specific layers.
/// </summary>
public class LayerwiseConfig
{
    /// <summary>
    /// Gets or sets the global default configuration.
    /// </summary>
    public QuantizationConfig GlobalConfig { get; set; }

    /// <summary>
    /// Gets or sets the per-layer configuration overrides.
    /// Key is the layer name, value is the layer-specific configuration.
    /// </summary>
    public Dictionary<string, LayerConfigEntry> LayerConfigs { get; set; }

    /// <summary>
    /// Gets the list of layer names with custom configurations.
    /// </summary>
    public IReadOnlyList<string> ConfiguredLayers => LayerConfigs.Keys.ToList().AsReadOnly();

    /// <summary>
    /// Creates a new LayerwiseConfig with the specified global configuration.
    /// </summary>
    /// <param name="globalConfig">The global default configuration.</param>
    public LayerwiseConfig(QuantizationConfig globalConfig)
    {
        GlobalConfig = globalConfig ?? throw new ArgumentNullException(nameof(globalConfig));
        LayerConfigs = new Dictionary<string, LayerConfigEntry>();
    }

    /// <summary>
    /// Creates a new LayerwiseConfig with a default global configuration.
    /// </summary>
    public LayerwiseConfig() : this(QuantizationConfig.CreateDefault())
    {
    }

    /// <summary>
    /// Sets a custom configuration for a specific layer.
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    /// <param name="config">The layer-specific configuration.</param>
    /// <param name="enabled">Whether quantization is enabled for this layer.</param>
    public void SetLayerConfig(string layerName, QuantizationConfig config, bool enabled = true)
    {
        if (string.IsNullOrEmpty(layerName))
            throw new ArgumentException("Layer name cannot be null or empty", nameof(layerName));

        if (config == null)
            throw new ArgumentNullException(nameof(config));

        LayerConfigs[layerName] = new LayerConfigEntry
        {
            LayerName = layerName,
            Config = config,
            Enabled = enabled
        };
    }

    /// <summary>
    /// Gets the configuration for a specific layer.
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    /// <returns>The layer configuration, or null if not found.</returns>
    public LayerConfigEntry? GetLayerConfig(string layerName)
    {
        if (string.IsNullOrEmpty(layerName))
            return null;

        return LayerConfigs.TryGetValue(layerName, out var config) ? config : null;
    }

    /// <summary>
    /// Gets the effective configuration for a layer.
    /// Returns the layer-specific configuration if available, otherwise returns the global configuration.
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    /// <returns>The effective configuration for the layer.</returns>
    public QuantizationConfig GetEffectiveConfig(string layerName)
    {
        var layerConfig = GetLayerConfig(layerName);
        return layerConfig?.Config ?? GlobalConfig;
    }

    /// <summary>
    /// Checks if quantization is enabled for a specific layer.
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    /// <returns>True if quantization is enabled for the layer.</returns>
    public bool IsLayerEnabled(string layerName)
    {
        var layerConfig = GetLayerConfig(layerName);
        return layerConfig?.Enabled ?? true; // Default to enabled
    }

    /// <summary>
    /// Disables quantization for a specific layer.
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    public void DisableLayer(string layerName)
    {
        if (LayerConfigs.ContainsKey(layerName))
        {
            LayerConfigs[layerName].Enabled = false;
        }
        else
        {
            LayerConfigs[layerName] = new LayerConfigEntry
            {
                LayerName = layerName,
                Config = GlobalConfig,
                Enabled = false
            };
        }
    }

    /// <summary>
    /// Enables quantization for a specific layer.
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    public void EnableLayer(string layerName)
    {
        if (LayerConfigs.ContainsKey(layerName))
        {
            LayerConfigs[layerName].Enabled = true;
        }
        else
        {
            LayerConfigs[layerName] = new LayerConfigEntry
            {
                LayerName = layerName,
                Config = GlobalConfig,
                Enabled = true
            };
        }
    }

    /// <summary>
    /// Sets the quantization mode for a specific layer.
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    /// <param name="weightMode">Weight quantization mode.</param>
    /// <param name="activationMode">Activation quantization mode.</param>
    public void SetLayerMode(string layerName, QuantizationMode weightMode, QuantizationMode activationMode)
    {
        var config = GetEffectiveConfig(layerName);
        config.WeightQuantization = weightMode;
        config.ActivationQuantization = activationMode;
        SetLayerConfig(layerName, config, IsLayerEnabled(layerName));
    }

    /// <summary>
    /// Removes a layer's custom configuration, reverting to the global configuration.
    /// </summary>
    /// <param name="layerName">Name of the layer.</param>
    /// <returns>True if the configuration was removed, false if it didn't exist.</returns>
    public bool RemoveLayerConfig(string layerName)
    {
        return LayerConfigs.Remove(layerName);
    }

    /// <summary>
    /// Clears all layer-specific configurations.
    /// </summary>
    public void ClearAllLayerConfigs()
    {
        LayerConfigs.Clear();
    }

    /// <summary>
    /// Gets all layer configurations.
    /// </summary>
    /// <returns>A dictionary of layer configurations.</returns>
    public Dictionary<string, QuantizationConfig> GetLayerConfigurationDictionary()
    {
        var dict = new Dictionary<string, QuantizationConfig>();
        foreach (var kvp in LayerConfigs)
        {
            if (kvp.Value.Enabled)
            {
                dict[kvp.Key] = kvp.Value.Config;
            }
        }
        return dict;
    }

    /// <summary>
    /// Gets the list of disabled layers.
    /// </summary>
    /// <returns>List of layer names with quantization disabled.</returns>
    public List<string> GetDisabledLayers()
    {
        return LayerConfigs
            .Where(kvp => !kvp.Value.Enabled)
            .Select(kvp => kvp.Key)
            .ToList();
    }

    /// <summary>
    /// Gets the list of enabled layers.
    /// </summary>
    /// <returns>List of layer names with quantization enabled.</returns>
    public List<string> GetEnabledLayers()
    {
        return LayerConfigs
            .Where(kvp => kvp.Value.Enabled)
            .Select(kvp => kvp.Key)
            .ToList();
    }

    /// <summary>
    /// Validates all layer configurations.
    /// </summary>
    /// <returns>True if all configurations are valid.</returns>
    /// <exception cref="InvalidOperationException">Thrown if any configuration is invalid.</exception>
    public bool Validate()
    {
        // Validate global config
        GlobalConfig.Validate();

        // Validate all layer configs
        foreach (var kvp in LayerConfigs)
        {
            kvp.Value.Config.Validate();
        }

        return true;
    }

    /// <summary>
    /// Creates a clone of this LayerwiseConfig.
    /// </summary>
    /// <returns>A deep copy of the configuration.</returns>
    public LayerwiseConfig Clone()
    {
        var clone = new LayerwiseConfig(CloneConfig(GlobalConfig));

        foreach (var kvp in LayerConfigs)
        {
            clone.LayerConfigs[kvp.Key] = new LayerConfigEntry
            {
                LayerName = kvp.Value.LayerName,
                Config = CloneConfig(kvp.Value.Config),
                Enabled = kvp.Value.Enabled
            };
        }

        return clone;
    }

    private static QuantizationConfig CloneConfig(QuantizationConfig config)
    {
        return new QuantizationConfig
        {
            WeightQuantization = config.WeightQuantization,
            ActivationQuantization = config.ActivationQuantization,
            CalibrationMethod = config.CalibrationMethod,
            CalibrationBatchSize = config.CalibrationBatchSize,
            QuantizationType = config.QuantizationType,
            FallbackToFP32 = config.FallbackToFP32,
            AccuracyThreshold = config.AccuracyThreshold,
            EnablePerChannelQuantization = config.EnablePerChannelQuantization,
            EnableMixedPrecision = config.EnableMixedPrecision
        };
    }
}

/// <summary>
/// Represents a layer-specific configuration entry.
/// </summary>
public class LayerConfigEntry
{
    /// <summary>
    /// Gets or sets the layer name.
    /// </summary>
    public string LayerName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the layer-specific configuration.
    /// </summary>
    public QuantizationConfig Config { get; set; } = null!;

    /// <summary>
    /// Gets or sets whether quantization is enabled for this layer.
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Returns a string representation of the layer config entry.
    /// </summary>
    public override string ToString()
    {
        return $"LayerConfigEntry({LayerName}, Enabled: {Enabled}, Config: {Config})";
    }
}
