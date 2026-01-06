namespace MLFramework.Optimizers.MixedPrecision;

public readonly struct PrecisionCapability
{
    public bool SupportsFP16 { get; init; }
    public bool SupportsBF16 { get; init; }
    public bool SupportsFP32 { get; init; }  // Always true

    public bool IsFP16Available => SupportsFP16;
    public bool IsBF16Available => SupportsBF16;
}
