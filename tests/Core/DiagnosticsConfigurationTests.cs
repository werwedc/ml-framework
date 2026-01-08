using Xunit;
using MLFramework.Core;

namespace MLFramework.Tests.Core;

public class DiagnosticsConfigurationTests
{
    #region Singleton Pattern Tests

    [Fact]
    public void Instance_ReturnsSameInstance()
    {
        // Arrange & Act
        var instance1 = DiagnosticsConfiguration.Instance;
        var instance2 = DiagnosticsConfiguration.Instance;

        // Assert
        Assert.Same(instance1, instance2);
    }

    [Fact]
    public void Instance_IsNotNull()
    {
        // Act
        var instance = DiagnosticsConfiguration.Instance;

        // Assert
        Assert.NotNull(instance);
    }

    #endregion

    #region Default Values Tests

    [Fact]
    public void DefaultValues_AreCorrect()
    {
        // Arrange
        var config = new DiagnosticsConfiguration();

        // Assert
        Assert.False(config.EnhancedErrorReporting);
        Assert.True(config.IncludeSuggestions);
        Assert.False(config.EnableShapeTracking);
        Assert.Equal(1, config.ContextTrackingDepth);
        Assert.False(config.DebugMode);
        Assert.False(config.LogShapeInformation);
    }

    #endregion

    #region Property Modification Tests

    [Fact]
    public void CanModifyEnhancedErrorReporting()
    {
        // Arrange
        var config = new DiagnosticsConfiguration();

        // Act
        config.EnhancedErrorReporting = true;

        // Assert
        Assert.True(config.EnhancedErrorReporting);
    }

    [Fact]
    public void CanModifyIncludeSuggestions()
    {
        // Arrange
        var config = new DiagnosticsConfiguration();

        // Act
        config.IncludeSuggestions = false;

        // Assert
        Assert.False(config.IncludeSuggestions);
    }

    [Fact]
    public void CanModifyEnableShapeTracking()
    {
        // Arrange
        var config = new DiagnosticsConfiguration();

        // Act
        config.EnableShapeTracking = true;

        // Assert
        Assert.True(config.EnableShapeTracking);
    }

    [Fact]
    public void CanModifyContextTrackingDepth()
    {
        // Arrange
        var config = new DiagnosticsConfiguration();

        // Act
        config.ContextTrackingDepth = 5;

        // Assert
        Assert.Equal(5, config.ContextTrackingDepth);
    }

    [Fact]
    public void CanModifyDebugMode()
    {
        // Arrange
        var config = new DiagnosticsConfiguration();

        // Act
        config.DebugMode = true;

        // Assert
        Assert.True(config.DebugMode);
    }

    [Fact]
    public void CanModifyLogShapeInformation()
    {
        // Arrange
        var config = new DiagnosticsConfiguration();

        // Act
        config.LogShapeInformation = true;

        // Assert
        Assert.True(config.LogShapeInformation);
    }

    #endregion

    #region ResetToDefaults Tests

    [Fact]
    public void ResetToDefaults_RestoresDefaultValues()
    {
        // Arrange
        var config = new DiagnosticsConfiguration();
        config.EnhancedErrorReporting = true;
        config.IncludeSuggestions = false;
        config.EnableShapeTracking = true;
        config.ContextTrackingDepth = 5;
        config.DebugMode = true;
        config.LogShapeInformation = true;

        // Act
        config.ResetToDefaults();

        // Assert
        Assert.False(config.EnhancedErrorReporting);
        Assert.True(config.IncludeSuggestions);
        Assert.False(config.EnableShapeTracking);
        Assert.Equal(1, config.ContextTrackingDepth);
        Assert.False(config.DebugMode);
        Assert.False(config.LogShapeInformation);
    }

    #endregion
}
