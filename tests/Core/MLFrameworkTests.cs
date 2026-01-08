using Xunit;
using MLFramework.Core;
using MLFramework;

namespace MLFramework.Tests.Core;

public class MLFrameworkTests
{
    #region Setup/Teardown

    public MLFrameworkTests()
    {
        // Ensure diagnostics are disabled before each test
        if (MLFramework.IsDiagnosticsEnabled())
        {
            MLFramework.DisableDiagnostics();
        }
    }

    #endregion

    #region EnableDiagnostics Tests

    [Fact]
    public void EnableDiagnostics_EnablesDiagnostics()
    {
        // Act
        MLFramework.EnableDiagnostics();

        // Assert
        Assert.True(MLFramework.IsDiagnosticsEnabled());
    }

    [Fact]
    public void EnableDiagnostics_SetsDefaultConfiguration()
    {
        // Arrange
        MLFramework.EnableDiagnostics();
        var config = MLFramework.GetDiagnosticsConfiguration();

        // Assert
        Assert.False(config.EnhancedErrorReporting);
        Assert.True(config.IncludeSuggestions);
        Assert.False(config.EnableShapeTracking);
        Assert.Equal(1, config.ContextTrackingDepth);
        Assert.False(config.DebugMode);
        Assert.False(config.LogShapeInformation);
    }

    [Fact]
    public void EnableDiagnostics_WithCustomConfig_UsesCustomSettings()
    {
        // Arrange
        var customConfig = new DiagnosticsConfiguration
        {
            EnhancedErrorReporting = true,
            IncludeSuggestions = false,
            EnableShapeTracking = true,
            ContextTrackingDepth = 3,
            DebugMode = true,
            LogShapeInformation = true
        };

        // Act
        MLFramework.EnableDiagnostics(customConfig);
        var config = MLFramework.GetDiagnosticsConfiguration();

        // Assert
        Assert.True(config.EnhancedErrorReporting);
        Assert.False(config.IncludeSuggestions);
        Assert.True(config.EnableShapeTracking);
        Assert.Equal(3, config.ContextTrackingDepth);
        Assert.True(config.DebugMode);
        Assert.True(config.LogShapeInformation);
    }

    [Fact]
    public void EnableDiagnostics_WithNullConfig_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => MLFramework.EnableDiagnostics(null!));
    }

    #endregion

    #region DisableDiagnostics Tests

    [Fact]
    public void DisableDiagnostics_DisablesDiagnostics()
    {
        // Arrange
        MLFramework.EnableDiagnostics();

        // Act
        MLFramework.DisableDiagnostics();

        // Assert
        Assert.False(MLFramework.IsDiagnosticsEnabled());
    }

    [Fact]
    public void DisableDiagnostics_ResetsConfigurationToDefaults()
    {
        // Arrange
        var customConfig = new DiagnosticsConfiguration
        {
            EnhancedErrorReporting = true,
            IncludeSuggestions = false,
            EnableShapeTracking = true,
            ContextTrackingDepth = 5,
            DebugMode = true,
            LogShapeInformation = true
        };
        MLFramework.EnableDiagnostics(customConfig);

        // Act
        MLFramework.DisableDiagnostics();
        var config = MLFramework.GetDiagnosticsConfiguration();

        // Assert
        Assert.False(config.EnhancedErrorReporting);
        Assert.True(config.IncludeSuggestions);
        Assert.False(config.EnableShapeTracking);
        Assert.Equal(1, config.ContextTrackingDepth);
        Assert.False(config.DebugMode);
        Assert.False(config.LogShapeInformation);
    }

    #endregion

    #region GetDiagnosticsConfiguration Tests

    [Fact]
    public void GetDiagnosticsConfiguration_ReturnsInstance()
    {
        // Act
        var config = MLFramework.GetDiagnosticsConfiguration();

        // Assert
        Assert.NotNull(config);
        Assert.IsType<DiagnosticsConfiguration>(config);
    }

    [Fact]
    public void GetDiagnosticsConfiguration_ReturnsSameInstance()
    {
        // Arrange
        MLFramework.EnableDiagnostics();

        // Act
        var config1 = MLFramework.GetDiagnosticsConfiguration();
        var config2 = MLFramework.GetDiagnosticsConfiguration();

        // Assert
        Assert.Same(config1, config2);
    }

    #endregion

    #region IsDiagnosticsEnabled Tests

    [Fact]
    public void IsDiagnosticsEnabled_InitiallyReturnsFalse()
    {
        // Act
        var isEnabled = MLFramework.IsDiagnosticsEnabled();

        // Assert
        Assert.False(isEnabled);
    }

    [Fact]
    public void IsDiagnosticsEnabled_AfterEnableReturnsTrue()
    {
        // Arrange
        MLFramework.EnableDiagnostics();

        // Act
        var isEnabled = MLFramework.IsDiagnosticsEnabled();

        // Assert
        Assert.True(isEnabled);
    }

    [Fact]
    public void IsDiagnosticsEnabled_AfterDisableReturnsFalse()
    {
        // Arrange
        MLFramework.EnableDiagnostics();
        MLFramework.DisableDiagnostics();

        // Act
        var isEnabled = MLFramework.IsDiagnosticsEnabled();

        // Assert
        Assert.False(isEnabled);
    }

    #endregion

    #region Configuration Persistence Tests

    [Fact]
    public void Configuration_PersistsAcrossOperations()
    {
        // Arrange
        var customConfig = new DiagnosticsConfiguration
        {
            EnhancedErrorReporting = true,
            EnableShapeTracking = true,
            ContextTrackingDepth = 2
        };
        MLFramework.EnableDiagnostics(customConfig);

        // Act
        var config1 = MLFramework.GetDiagnosticsConfiguration();
        var config2 = MLFramework.GetDiagnosticsConfiguration();

        // Assert
        Assert.Equal(config1.EnhancedErrorReporting, config2.EnhancedErrorReporting);
        Assert.Equal(config1.EnableShapeTracking, config2.EnableShapeTracking);
        Assert.Equal(config1.ContextTrackingDepth, config2.ContextTrackingDepth);
    }

    [Fact]
    public void MultipleEnableDiagnosticsCalls_UseLatestConfiguration()
    {
        // Arrange
        var config1 = new DiagnosticsConfiguration { EnhancedErrorReporting = true };
        var config2 = new DiagnosticsConfiguration { EnhancedErrorReporting = false };

        // Act
        MLFramework.EnableDiagnostics(config1);
        MLFramework.EnableDiagnostics(config2);
        var finalConfig = MLFramework.GetDiagnosticsConfiguration();

        // Assert
        Assert.False(finalConfig.EnhancedErrorReporting);
    }

    #endregion

    #region Convenience Methods Tests

    [Fact]
    public void EnableEnhancedErrorReporting_EnablesDiagnosticsAndEnhancedReporting()
    {
        // Act
        MLFramework.EnableEnhancedErrorReporting();
        var config = MLFramework.GetDiagnosticsConfiguration();

        // Assert
        Assert.True(MLFramework.IsDiagnosticsEnabled());
        Assert.True(config.EnhancedErrorReporting);
    }

    [Fact]
    public void EnableShapeTracking_EnablesDiagnosticsAndShapeTracking()
    {
        // Act
        MLFramework.EnableShapeTracking();
        var config = MLFramework.GetDiagnosticsConfiguration();

        // Assert
        Assert.True(MLFramework.IsDiagnosticsEnabled());
        Assert.True(config.EnableShapeTracking);
    }

    [Fact]
    public void EnableDebugMode_EnablesDiagnosticsAndDebugMode()
    {
        // Act
        MLFramework.EnableDebugMode();
        var config = MLFramework.GetDiagnosticsConfiguration();

        // Assert
        Assert.True(MLFramework.IsDiagnosticsEnabled());
        Assert.True(config.DebugMode);
        Assert.True(config.LogShapeInformation);
    }

    #endregion

    #region Performance Tests

    [Fact]
    public void Performance_DiagnosticsDisabled_MinimalOverhead()
    {
        // Arrange
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        // Act
        for (int i = 0; i < 1000; i++)
        {
            var enabled = MLFramework.IsDiagnosticsEnabled();
        }
        stopwatch.Stop();

        // Assert
        // Should complete in under 10ms even on slower machines
        Assert.True(stopwatch.ElapsedMilliseconds < 10,
            $"Diagnostics check took {stopwatch.ElapsedMilliseconds}ms, expected < 10ms");
    }

    #endregion
}
