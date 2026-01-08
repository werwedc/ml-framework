namespace MLFramework.Communication.Tests.Backends;

using MLFramework.Communication.Backends;
using Xunit;

/// <summary>
/// Unit tests for MPI configuration
/// </summary>
public class MPIConfigTests
{
    [Fact]
    public void MPIConfig_DefaultValues_AreCorrect()
    {
        // Arrange
        var config = new MPIConfig();

        // Assert
        Assert.True(config.UseCudaAwareMPI);
        Assert.Equal(1, config.ThreadLevel);
        Assert.False(config.EnableProfiling);
        Assert.Equal(65536, config.BufferSize);
        Assert.False(config.EnableTuning);
    }

    [Fact]
    public void MPIConfig_CanModifyProperties()
    {
        // Arrange
        var config = new MPIConfig();

        // Act
        config.UseCudaAwareMPI = false;
        config.ThreadLevel = 2;
        config.EnableProfiling = true;
        config.BufferSize = 131072;
        config.EnableTuning = true;

        // Assert
        Assert.False(config.UseCudaAwareMPI);
        Assert.Equal(2, config.ThreadLevel);
        Assert.True(config.EnableProfiling);
        Assert.Equal(131072, config.BufferSize);
        Assert.True(config.EnableTuning);
    }

    [Fact]
    public void MPIConfig_Apply_SetsEnvironmentVariables()
    {
        // Arrange
        var config = new MPIConfig
        {
            EnableProfiling = true,
            EnableTuning = true
        };

        // Act
        config.Apply();

        // Assert
        var stats = Environment.GetEnvironmentVariable("I_MPI_STATS");
        var allreduce = Environment.GetEnvironmentVariable("I_MPI_ADJUST_ALLREDUCE");

        Assert.Equal("1", stats);
        Assert.Equal("2", allreduce);
    }

    [Fact]
    public void MPIConfig_Apply_WhenNotProfiling_SetsWarn()
    {
        // Arrange
        var config = new MPIConfig
        {
            EnableProfiling = false
        };

        // Act
        config.Apply();

        // Assert
        var stats = Environment.GetEnvironmentVariable("I_MPI_STATS");
        Assert.Null(stats);
    }
}
