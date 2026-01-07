namespace MLFramework.Fusion;

/// <summary>
/// Factory for creating and configuring fusion registries
/// </summary>
public static class FusionRegistryFactory
{
    /// <summary>
    /// Creates a default fusion registry with all standard patterns
    /// </summary>
    /// <returns>A configured IFusionRegistry instance</returns>
    public static IFusionRegistry CreateDefault()
    {
        return new DefaultFusionRegistry();
    }

    /// <summary>
    /// Creates a fusion registry and applies custom configuration
    /// </summary>
    /// <param name="configure">Configuration action</param>
    /// <returns>A configured IFusionRegistry instance</returns>
    public static IFusionRegistry CreateWithCustomPatterns(
        Action<IFusionRegistry> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);

        var registry = new DefaultFusionRegistry();
        configure(registry);
        return registry;
    }

    /// <summary>
    /// Creates an empty fusion registry (no default patterns)
    /// </summary>
    /// <returns>An empty IFusionRegistry instance</returns>
    public static IFusionRegistry CreateEmpty()
    {
        return new DefaultFusionRegistry(skipDefaults: true);
    }
}
