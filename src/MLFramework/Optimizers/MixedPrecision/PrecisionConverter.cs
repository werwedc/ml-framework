using System;

namespace MLFramework.Optimizers.MixedPrecision;

public static class PrecisionConverter
{
    /// <summary>
    /// Converts tensor to target precision
    /// </summary>
    public static ITensor Convert(ITensor tensor, Precision targetPrecision)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var currentPrecision = DetectPrecision(tensor);

        if (currentPrecision == targetPrecision)
            return tensor;

        // TODO: Implement actual conversion
        // For now, return tensor as-is (will be implemented in subsequent specs)
        return tensor;
    }

    /// <summary>
    /// Detects the precision of a tensor
    /// </summary>
    public static Precision DetectPrecision(ITensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // TODO: Implement based on tensor dtype
        // For now, return FP32 as default
        return Precision.FP32;
    }

    /// <summary>
    /// Checks if conversion is safe (no significant precision loss)
    /// </summary>
    public static bool IsConversionSafe(Precision from, Precision to)
    {
        // FP32 -> FP16/BF16: May lose precision but generally safe
        // FP16/BF16 -> FP32: Always safe (no loss)
        // FP16 <-> BF16: May have precision differences

        if (from == to)
            return true;

        if (to == Precision.FP32)
            return true;

        // Converting to reduced precision
        return true;  // Conservative: allow all conversions
    }
}

/// <summary>
/// Interface for tensor operations (will be defined in tensor infrastructure)
/// </summary>
public interface ITensor
{
    /// <summary>
    /// Gets the precision of the tensor
    /// </summary>
    Precision Precision { get; }
}
