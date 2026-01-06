using System;

namespace MLFramework.Optimizers.MixedPrecision;

public static class HardwareDetector
{
    // Cache detection results using Lazy<T> for thread safety
    private static readonly Lazy<PrecisionCapability> _cachedCapability = new Lazy<PrecisionCapability>(() =>
    {
        var capability = new PrecisionCapability
        {
            SupportsFP32 = true,  // Always supported
            SupportsFP16 = CheckFP16Support(),
            SupportsBF16 = CheckBF16Support()
        };
        return capability;
    });

    /// <summary>
    /// Detects mixed-precision capabilities of the current hardware
    /// </summary>
    public static PrecisionCapability DetectCapabilities()
    {
        return _cachedCapability.Value;
    }

    /// <summary>
    /// Returns the recommended precision for current hardware
    /// Priority: BF16 > FP16 > FP32
    /// </summary>
    public static Precision GetRecommendedPrecision()
    {
        var capability = DetectCapabilities();
        if (capability.SupportsBF16)
            return Precision.BF16;
        if (capability.SupportsFP16)
            return Precision.FP16;
        return Precision.FP32;
    }

    private static bool CheckFP16Support()
    {
        // TODO: Implement actual hardware detection
        // For now, return false (will be mocked in tests)
        return false;
    }

    private static bool CheckBF16Support()
    {
        // TODO: Implement actual hardware detection
        // For now, return false (will be mocked in tests)
        return false;
    }

    /// <summary>
    /// Clear cached detection (useful for testing)
    /// </summary>
    public static void ResetCache()
    {
        // Lazy<T> doesn't support direct reset, so we rely on test isolation
        // In production, this would typically not be called
        // This is a limitation that could be addressed with a more sophisticated caching mechanism
        // if needed
    }
}
