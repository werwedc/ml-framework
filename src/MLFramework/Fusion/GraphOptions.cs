namespace MLFramework.Fusion;

/// <summary>
/// Configuration options for controlling fusion behavior globally
/// </summary>
public static class GraphOptions
{
    /// <summary>
    /// Enables or disables fusion globally
    /// </summary>
    public static bool EnableFusion { get; set; } = true;

    /// <summary>
    /// Maximum number of operations to fuse together
    /// </summary>
    public static int MaxFusionOps { get; set; } = 10;

    /// <summary>
    /// Fusion backend to use
    /// </summary>
    public static FusionBackend FusionBackend { get; set; } = FusionBackend.Triton;

    /// <summary>
    /// Minimum benefit score required to apply fusion
    /// </summary>
    public static int MinBenefitScore { get; set; } = 50;

    /// <summary>
    /// Fusion aggressiveness level
    /// </summary>
    public static FusionAggressiveness Aggressiveness { get; set; } = FusionAggressiveness.Medium;

    /// <summary>
    /// Enable automatic fusion (default behavior)
    /// </summary>
    public static bool EnableAutomaticFusion { get; set; } = true;

    /// <summary>
    /// Enable fusion with user hints
    /// </summary>
    public static bool EnableHintedFusion { get; set; } = true;

    /// <summary>
    /// Enable autotuning for fused kernels
    /// </summary>
    public static bool EnableAutotuning { get; set; } = true;

    /// <summary>
    /// Enable BatchNorm folding into convolutions
    /// </summary>
    public static bool EnableBatchNormFolding { get; set; } = true;

    /// <summary>
    /// Enable Conv+Activation fusion
    /// </summary>
    public static bool EnableConvActivationFusion { get; set; } = true;

    /// <summary>
    /// Enable element-wise operation fusion
    /// </summary>
    public static bool EnableElementWiseFusion { get; set; } = true;

    /// <summary>
    /// Cache directory for autotuning results
    /// </summary>
    public static string? TuningCacheDirectory { get; set; }

    /// <summary>
    /// Resets all options to default values
    /// </summary>
    public static void ResetToDefaults()
    {
        EnableFusion = true;
        MaxFusionOps = 10;
        FusionBackend = FusionBackend.Triton;
        MinBenefitScore = 50;
        Aggressiveness = FusionAggressiveness.Medium;
        EnableAutomaticFusion = true;
        EnableHintedFusion = true;
        EnableAutotuning = true;
        EnableBatchNormFolding = true;
        EnableConvActivationFusion = true;
        EnableElementWiseFusion = true;
        TuningCacheDirectory = null;
    }
}
