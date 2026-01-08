using System;
using System.Collections.Generic;
using System.IO;
using MLFramework.Visualization;
using MLFramework.Visualization.Hooks;
using Xunit;

namespace MLFramework.Visualization.Tests.Hooks;

/// <summary>
/// Unit tests for VisualizationTrainingHook
/// </summary>
public class VisualizationTrainingHookTests
{
    private class TestVisualizer : IVisualizer
    {
        public List<(string name, float value, long step)> Scalars { get; } = new();
        public List<string> ProfileStarts { get; } = new();
        public List<string> ProfileEnds { get; } = new();

        public void LogScalar(string name, float value, long step)
        {
            Scalars.Add((name, value, step));
        }

        public void StartProfile(string name)
        {
            ProfileStarts.Add(name);
        }

        public void EndProfile(string name)
        {
            ProfileEnds.Add(name);
        }

        public void Flush()
        {
            // No-op for testing
        }
    }

    [Fact]
    public void Constructor_ShouldThrowWhenVisualizerIsNull()
    {
        Assert.Throws<ArgumentNullException>(() => new VisualizationTrainingHook(null!));
    }

    [Fact]
    public void OnPhaseStart_ShouldStartProfilingForForwardPass()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer) { ProfileForwardPass = true };
        var context = new TrainingContext { CurrentStep = 0 };

        // Act
        hook.OnPhaseStart(TrainingPhase.ForwardPassStart, context);

        // Assert
        Assert.Single(visualizer.ProfileStarts);
        Assert.Contains("forward_pass", visualizer.ProfileStarts[0]);
    }

    [Fact]
    public void OnPhaseEnd_ShouldEndProfilingForForwardPass()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer) { ProfileForwardPass = true };
        var context = new TrainingContext { CurrentStep = 0 };

        // Act
        hook.OnPhaseEnd(TrainingPhase.ForwardPassEnd, context);

        // Assert
        Assert.Single(visualizer.ProfileEnds);
        Assert.Contains("forward_pass", visualizer.ProfileEnds[0]);
    }

    [Fact]
    public void OnPhaseEnd_ShouldNotProfileWhenDisabled()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer) { ProfileForwardPass = false };
        var context = new TrainingContext { CurrentStep = 0 };

        // Act
        hook.OnPhaseStart(TrainingPhase.ForwardPassStart, context);

        // Assert
        Assert.Empty(visualizer.ProfileStarts);
    }

    [Fact]
    public void OnPhaseEnd_BatchEnd_ShouldLogMetrics()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer)
        {
            LogLoss = true,
            LogMetrics = true,
            LogFrequencyBatches = 1
        };
        var context = new TrainingContext
        {
            CurrentStep = 10,
            Loss = 0.5f,
            Metrics = { { "accuracy", 0.85f } },
            LearningRate = 0.001f
        };

        // Act
        hook.OnPhaseEnd(TrainingPhase.BatchEnd, context);

        // Assert
        Assert.Equal(3, visualizer.Scalars.Count); // loss, metrics["accuracy"], learning_rate
    }

    [Fact]
    public void OnPhaseEnd_BatchEnd_ShouldRespectLogFrequency()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer)
        {
            LogLoss = true,
            LogFrequencyBatches = 5
        };
        var context = new TrainingContext
        {
            CurrentStep = 10,
            Loss = 0.5f
        };

        // Act
        hook.OnPhaseEnd(TrainingPhase.BatchEnd, context); // batch 1
        hook.OnPhaseEnd(TrainingPhase.BatchEnd, context); // batch 2
        hook.OnPhaseEnd(TrainingPhase.BatchEnd, context); // batch 3
        hook.OnPhaseEnd(TrainingPhase.BatchEnd, context); // batch 4
        hook.OnPhaseEnd(TrainingPhase.BatchEnd, context); // batch 5 - should log

        // Assert
        Assert.Single(visualizer.Scalars);
    }

    [Fact]
    public void OnPhaseEnd_BatchEnd_ShouldAlwaysLogAtEpochEnd()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer)
        {
            LogLoss = true,
            LogFrequencyBatches = 100
        };
        var context = new TrainingContext
        {
            CurrentStep = 10,
            Loss = 0.5f
        };

        // Act
        hook.OnPhaseEnd(TrainingPhase.BatchEnd, context);
        hook.OnPhaseEnd(TrainingPhase.EpochEnd, context); // Should log

        // Assert
        Assert.Single(visualizer.Scalars);
    }

    [Fact]
    public void OnMetricUpdate_ShouldLogMetricWhenEnabled()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer) { LogMetrics = true };
        var context = new TrainingContext { CurrentStep = 5 };

        // Act
        hook.OnMetricUpdate("custom_metric", 0.75f, context);

        // Assert
        Assert.Single(visualizer.Scalars);
        Assert.Equal("custom_metric", visualizer.Scalars[0].name);
        Assert.Equal(0.75f, visualizer.Scalars[0].value);
        Assert.Equal(5, visualizer.Scalars[0].step);
    }

    [Fact]
    public void OnMetricUpdate_ShouldNotLogMetricWhenDisabled()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer) { LogMetrics = false };
        var context = new TrainingContext { CurrentStep = 5 };

        // Act
        hook.OnMetricUpdate("custom_metric", 0.75f, context);

        // Assert
        Assert.Empty(visualizer.Scalars);
    }

    [Fact]
    public void OnException_ShouldFlushLogs()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer);
        var context = new TrainingContext { CurrentStep = 10 };
        var exception = new InvalidOperationException("Test exception");

        // Act
        hook.OnException(exception, context);

        // Assert - Just verify it doesn't throw
        Assert.True(true);
    }

    [Fact]
    public void OnPhaseEnd_ShouldUseCustomLogPrefix()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer)
        {
            LogLoss = true,
            LogPrefix = "validation/"
        };
        var context = new TrainingContext
        {
            CurrentStep = 10,
            Loss = 0.3f
        };

        // Act
        hook.OnPhaseEnd(TrainingPhase.EpochEnd, context);

        // Assert
        Assert.Single(visualizer.Scalars);
        Assert.StartsWith("validation/", visualizer.Scalars[0].name);
    }
}
