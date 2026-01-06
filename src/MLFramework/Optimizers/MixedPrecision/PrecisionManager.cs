using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace MLFramework.Optimizers.MixedPrecision;

/// <summary>
/// Manages tensor precision conversions and layer exclusion rules
/// </summary>
public class PrecisionManager
{
    private readonly MixedPrecisionOptions _options;
    private readonly HashSet<string> _fp32ExcludedLayers;
    private readonly Precision _targetPrecision;
    private readonly Dictionary<string, Regex> _compiledRegexCache;

    #region Properties

    /// <summary>
    /// Target training precision
    /// </summary>
    public Precision TargetPrecision => _targetPrecision;

    /// <summary>
    /// Whether the manager is in reduced precision mode
    /// </summary>
    public bool IsReducedPrecision => _targetPrecision != Precision.FP32;

    /// <summary>
    /// Number of layers excluded from mixed precision
    /// </summary>
    public int ExcludedLayerCount => _fp32ExcludedLayers.Count;

    #endregion

    #region Constructors

    public PrecisionManager(MixedPrecisionOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        options.Validate();

        _targetPrecision = options.AutoDetectPrecision
            ? HardwareDetector.GetRecommendedPrecision()
            : options.Precision;

        _fp32ExcludedLayers = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        _compiledRegexCache = new Dictionary<string, Regex>();

        if (options.AutoExcludeSensitiveLayers)
        {
            AddDefaultExclusions();
        }

        // Add custom exclusions
        foreach (var pattern in options.Fp32LayerPatterns)
        {
            ExcludeLayersMatching(pattern);
        }
    }

    public PrecisionManager()
        : this(MixedPrecisionOptions.ForFP16())
    {
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Converts a tensor to training precision (FP16/BF16) unless excluded
    /// </summary>
    public ITensor ConvertToTrainingPrecision(ITensor tensor, string? layerName = null)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // If layer is excluded, keep in FP32
        if (layerName != null && ShouldExcludeLayer(layerName))
            return tensor;

        // If already in target precision, no conversion needed
        var currentPrecision = PrecisionConverter.DetectPrecision(tensor);
        if (currentPrecision == _targetPrecision)
            return tensor;

        // Convert to target precision
        return PrecisionConverter.Convert(tensor, _targetPrecision);
    }

    /// <summary>
    /// Converts a tensor to FP32 precision (master weights)
    /// </summary>
    public ITensor ConvertToFP32(ITensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var currentPrecision = PrecisionConverter.DetectPrecision(tensor);
        if (currentPrecision == Precision.FP32)
            return tensor;

        return PrecisionConverter.Convert(tensor, Precision.FP32);
    }

    /// <summary>
    /// Converts a dictionary of tensors to target precision
    /// </summary>
    public Dictionary<string, ITensor> ConvertWeights(Dictionary<string, ITensor> weights)
    {
        if (weights == null)
            throw new ArgumentNullException(nameof(weights));

        var converted = new Dictionary<string, ITensor>();
        foreach (var kvp in weights)
        {
            converted[kvp.Key] = ConvertToTrainingPrecision(kvp.Value, kvp.Key);
        }

        return converted;
    }

    /// <summary>
    /// Converts a dictionary of tensors to FP32
    /// </summary>
    public Dictionary<string, ITensor> ConvertToFP32(Dictionary<string, ITensor> tensors)
    {
        if (tensors == null)
            throw new ArgumentNullException(nameof(tensors));

        var converted = new Dictionary<string, ITensor>();
        foreach (var kvp in tensors)
        {
            converted[kvp.Key] = ConvertToFP32(kvp.Value);
        }

        return converted;
    }

    /// <summary>
    /// Checks if a layer should be excluded from mixed precision
    /// </summary>
    public bool ShouldExcludeLayer(string layerName)
    {
        if (string.IsNullOrEmpty(layerName))
            return false;

        foreach (var pattern in _fp32ExcludedLayers)
        {
            if (IsMatch(layerName, pattern))
                return true;
        }

        return false;
    }

    /// <summary>
    /// Adds a pattern to exclude layers from mixed precision
    /// </summary>
    public void ExcludeLayersMatching(string pattern)
    {
        if (string.IsNullOrWhiteSpace(pattern))
            throw new ArgumentException("Pattern cannot be empty", nameof(pattern));

        _fp32ExcludedLayers.Add(pattern);
    }

    /// <summary>
    /// Removes a pattern from exclusions
    /// </summary>
    public bool RemoveExclusion(string pattern)
    {
        _compiledRegexCache.Remove(pattern);
        return _fp32ExcludedLayers.Remove(pattern);
    }

    /// <summary>
    /// Clears all layer exclusions
    /// </summary>
    public void ClearExclusions()
    {
        _fp32ExcludedLayers.Clear();
        _compiledRegexCache.Clear();
    }

    /// <summary>
    /// Gets all current exclusion patterns
    /// </summary>
    public IEnumerable<string> GetExclusionPatterns()
    {
        return _fp32ExcludedLayers.ToList();
    }

    /// <summary>
    /// Creates a master FP32 copy of training weights
    /// </summary>
    public Dictionary<string, ITensor> CreateMasterWeights(Dictionary<string, ITensor> trainingWeights)
    {
        return ConvertToFP32(trainingWeights);
    }

    /// <summary>
    /// Syncs training weights from master weights (after optimizer update)
    /// </summary>
    public Dictionary<string, ITensor> SyncTrainingWeights(Dictionary<string, ITensor> masterWeights)
    {
        return ConvertWeights(masterWeights);
    }

    #endregion

    #region Private Methods

    private void AddDefaultExclusions()
    {
        // Add common sensitive layer types
        var defaults = new[]
        {
            "BatchNorm",
            "LayerNorm",
            "InstanceNorm",
            "GroupNorm",
            "Embedding"  // Some models benefit from FP32 embeddings
        };

        foreach (var pattern in defaults)
        {
            _fp32ExcludedLayers.Add(pattern);
        }
    }

    private bool IsMatch(string layerName, string pattern)
    {
        // Try exact match first
        if (layerName.Equals(pattern, StringComparison.OrdinalIgnoreCase))
            return true;

        // Try contains match (case-insensitive)
        if (layerName.IndexOf(pattern, StringComparison.OrdinalIgnoreCase) >= 0)
            return true;

        // Try regex match (if pattern contains regex metacharacters)
        try
        {
            if (!_compiledRegexCache.TryGetValue(pattern, out var regex))
            {
                regex = new Regex(pattern, RegexOptions.IgnoreCase | RegexOptions.Compiled);
                _compiledRegexCache[pattern] = regex;
            }

            return regex.IsMatch(layerName);
        }
        catch (ArgumentException)
        {
            // Invalid regex, skip
            return false;
        }
    }

    #endregion
}

/// <summary>
/// Statistics about precision conversion
/// </summary>
public class PrecisionConversionStats
{
    public int TotalConversions { get; set; }
    public int TrainingPrecisionConversions { get; set; }
    public int FP32Conversions { get; set; }
    public int SkippedConversions { get; set; }  // Already in target precision
    public int ExcludedLayers { get; set; }

    public override string ToString()
    {
        return $"Total: {TotalConversions}, " +
               $"Training: {TrainingPrecisionConversions}, " +
               $"FP32: {FP32Conversions}, " +
               $"Skipped: {SkippedConversions}, " +
               $"Excluded: {ExcludedLayers}";
    }
}
