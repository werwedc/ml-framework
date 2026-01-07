namespace MLFramework.Fusion;

/// <summary>
/// Context manager for temporarily modifying fusion options
/// </summary>
public sealed class FusionContext : IDisposable
{
    private readonly FusionOptions _previousOptions;
    private readonly bool _restoreOnDispose;

    /// <summary>
    /// Creates a context with custom fusion options
    /// </summary>
    /// <param name="options">Fusion options to apply</param>
    /// <param name="restoreOnDispose">Whether to restore previous options on dispose</param>
    public FusionContext(FusionOptions options, bool restoreOnDispose = true)
    {
        _previousOptions = GetCurrentOptions();
        _restoreOnDispose = restoreOnDispose;
        SetCurrentOptions(options);
    }

    /// <summary>
    /// Restores previous fusion options when disposed
    /// </summary>
    public void Dispose()
    {
        if (_restoreOnDispose)
        {
            SetCurrentOptions(_previousOptions);
        }
    }

    private static FusionOptions GetCurrentOptions()
    {
        return new FusionOptions
        {
            EnableFusion = GraphOptions.EnableFusion,
            MaxFusionOps = GraphOptions.MaxFusionOps,
            FusionBackend = GraphOptions.FusionBackend,
            MinBenefitScore = GraphOptions.MinBenefitScore,
            Aggressiveness = GraphOptions.Aggressiveness
        };
    }

    private static void SetCurrentOptions(FusionOptions options)
    {
        GraphOptions.EnableFusion = options.EnableFusion;
        GraphOptions.MaxFusionOps = options.MaxFusionOps;
        GraphOptions.FusionBackend = options.FusionBackend;
        GraphOptions.MinBenefitScore = options.MinBenefitScore;
        GraphOptions.Aggressiveness = options.Aggressiveness;
    }

    /// <summary>
    /// Creates a context with fusion disabled
    /// </summary>
    public static FusionContext DisableFusion()
    {
        return new FusionContext(new FusionOptions { EnableFusion = false });
    }

    /// <summary>
    /// Creates a context with custom fusion options
    /// </summary>
    /// <param name="configure">Action to configure fusion options</param>
    public static FusionContext WithOptions(Action<FusionOptions> configure)
    {
        var options = GetCurrentOptions();
        configure(options);
        return new FusionContext(options);
    }
}
