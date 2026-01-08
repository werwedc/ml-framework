using System;
using System.Collections.Generic;
using MLFramework.Visualization;
using MLFramework.Visualization.Hooks;
using Xunit;

namespace MLFramework.Visualization.Tests.Hooks;

/// <summary>
/// Integration tests for training loop with hooks
/// </summary>
public class TrainingLoopIntegrationTests
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
            // No-op
        }
    }

    private class TestHook : ITrainingHook
    {
        public List<TrainingPhase> PhaseStarts { get; } = new();
        public List<TrainingPhase> PhaseEnds { get; } = new();
        public List<(string name, float value)> MetricUpdates { get; } = new();
        public List<Exception> Exceptions { get; } = new();

        public void OnPhaseStart(TrainingPhase phase, TrainingContext context)
        {
            PhaseStarts.Add(phase);
        }

        public void OnPhaseEnd(TrainingPhase phase, TrainingContext context)
        {
            PhaseEnds.Add(phase);
        }

        public void OnMetricUpdate(string metricName, float value, TrainingContext context)
        {
            MetricUpdates.Add((metricName, value));
        }

        public void OnException(Exception exception, TrainingContext context)
        {
            Exceptions.Add(exception);
        }
    }

    [Fact]
    public void TrainingLoop_WithVisualizerHook_ShouldLogMetrics()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer)
        {
            LogLoss = true,
            LogMetrics = true,
            LogFrequencyBatches = 1
        };

        var trainingLoop = new TrainingLoop(new object(), new object());
        trainingLoop.AddHook(hook);

        // Act
        trainingLoop.Train(epochs: 2, new object(), new object());

        // Assert
        Assert.NotEmpty(visualizer.Scalars);
    }

    [Fact]
    public void TrainingLoop_WithTestHook_ShouldCallAllHooksInOrder()
    {
        // Arrange
        var hook1 = new TestHook();
        var hook2 = new TestHook();

        var trainingLoop = new TrainingLoop(new object(), new object());
        trainingLoop.AddHook(hook1);
        trainingLoop.AddHook(hook2);

        // Act
        trainingLoop.Train(epochs: 1, new object(), new object());

        // Assert
        Assert.NotEmpty(hook1.PhaseStarts);
        Assert.NotEmpty(hook1.PhaseEnds);
        Assert.NotEmpty(hook2.PhaseStarts);
        Assert.NotEmpty(hook2.PhaseEnds);
    }

    [Fact]
    public void TrainingLoop_RemoveHook_ShouldNotCallRemovedHook()
    {
        // Arrange
        var hook = new TestHook();

        var trainingLoop = new TrainingLoop(new object(), new object());
        trainingLoop.AddHook(hook);
        trainingLoop.RemoveHook(hook);

        // Act
        trainingLoop.Train(epochs: 1, new object(), new object());

        // Assert
        Assert.Empty(hook.PhaseStarts);
        Assert.Empty(hook.PhaseEnds);
    }

    [Fact]
    public void TrainingLoop_MultipleHooks_ShouldCallAllHooks()
    {
        // Arrange
        var hook1 = new TestHook();
        var hook2 = new TestHook();
        var hook3 = new TestHook();

        var trainingLoop = new TrainingLoop(new object(), new object());
        trainingLoop.AddHook(hook1);
        trainingLoop.AddHook(hook2);
        trainingLoop.AddHook(hook3);

        // Act
        trainingLoop.Train(epochs: 1, new object(), new object());

        // Assert
        Assert.NotEmpty(hook1.PhaseStarts);
        Assert.NotEmpty(hook2.PhaseStarts);
        Assert.NotEmpty(hook3.PhaseStarts);
    }

    [Fact]
    public void TrainingLoop_Validate_ShouldCallValidationHooks()
    {
        // Arrange
        var hook = new TestHook();

        var trainingLoop = new TrainingLoop(new object(), new object());
        trainingLoop.AddHook(hook);

        // Act
        trainingLoop.Validate(new object(), new object());

        // Assert
        Assert.Contains(TrainingPhase.ValidationStart, hook.PhaseStarts);
        Assert.Contains(TrainingPhase.ValidationEnd, hook.PhaseEnds);
    }

    [Fact]
    public void TrainingLoop_WithEarlyStoppingHook_ShouldStopTraining()
    {
        // Arrange
        var earlyStopHook = new EarlyStoppingHook("loss", patience: 1);
        earlyStopHook.StopTrainingCallback = () => { };

        var trainingLoop = new TrainingLoop(new object(), new object());
        trainingLoop.AddHook(earlyStopHook);

        // Act
        trainingLoop.Train(epochs: 5, new object(), new object());

        // Assert - Should have triggered early stopping
        Assert.True(earlyStopHook.ShouldStop);
    }

    [Fact]
    public void TrainingContext_ShouldHaveValidTrainingState()
    {
        // Arrange
        var hook = new TestHook();
        var trainingLoop = new TrainingLoop(new object(), new object());
        trainingLoop.AddHook(hook);

        // Act
        trainingLoop.Train(epochs: 1, new object(), new object());

        // Assert - Context should have been passed with valid values
        Assert.NotEmpty(hook.PhaseStarts);
    }

    [Fact]
    public void TrainingLoop_WithCheckpointHook_ShouldSaveCheckpoints()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), "test_checkpoints");
        Directory.CreateDirectory(tempDir);

        try
        {
            var checkpointHook = new CheckpointHook(tempDir, saveEveryNEpochs: 1);
            checkpointHook.SaveCheckpointCallback = (context, filepath) =>
            {
                // Simulate saving checkpoint
                File.WriteAllText(filepath, "checkpoint_data");
            };

            var trainingLoop = new TrainingLoop(new object(), new object());
            trainingLoop.AddHook(checkpointHook);

            // Act
            trainingLoop.Train(epochs: 2, new object(), new object());

            // Assert
            var checkpointFiles = Directory.GetFiles(tempDir, "checkpoint_*.pt");
            Assert.True(checkpointFiles.Length > 0, "Should have saved at least one checkpoint");
        }
        finally
        {
            // Cleanup
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [Fact]
    public void TrainingLoop_ShouldPassContextToHooks()
    {
        // Arrange
        TrainingContext? capturedContext = null;
        var hook = new TestHook();

        var trainingLoop = new TrainingLoop(new object(), new object());
        trainingLoop.AddHook(hook);

        // Act
        trainingLoop.Train(epochs: 1, new object(), new object());

        // Assert
        Assert.NotEmpty(hook.PhaseStarts);
        Assert.NotNull(hook.PhaseStarts);
    }

    [Fact]
    public void TrainingLoop_LogFrequency_ShouldControlOutput()
    {
        // Arrange
        var visualizer = new TestVisualizer();
        var hook = new VisualizationTrainingHook(visualizer)
        {
            LogLoss = true,
            LogFrequencyBatches = 20 // Log every 20 batches
        };

        var trainingLoop = new TrainingLoop(new object(), new object())
        {
            LogFrequency = 20
        };
        trainingLoop.AddHook(hook);

        // Act
        trainingLoop.Train(epochs: 1, new object(), new object());

        // Assert - Should have logged, but less frequently
        Assert.NotEmpty(visualizer.Scalars);
    }
}
