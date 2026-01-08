using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MachineLearning.Checkpointing;
using MachineLearning.Checkpointing.Async;
using Xunit;

namespace MLFramework.Tests.Checkpointing.Async;

/// <summary>
/// Tests for async checkpointing functionality
/// </summary>
public class AsyncCheckpointTests : IDisposable
{
    private readonly string _testCheckpointDir;
    private readonly string _tempDir;

    public AsyncCheckpointTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"async_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDir);
        _testCheckpointDir = Path.Combine(_tempDir, "checkpoint");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            try
            {
                Directory.Delete(_tempDir, true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    [Fact]
    public void Constructor_WithValidParameters_CreatesManager()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);

        // Act
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5);

        // Assert
        Assert.Equal(5, manager.MaxQueueSize);
        Assert.Equal(0, manager.QueueSize);
        Assert.Equal(0, manager.ActiveTaskCount);
    }

    [Fact]
    public async Task QueueSaveAsync_WithValidParameters_QueuesTaskSuccessfully()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5);
        var model = new TestModel();
        var optimizer = new TestOptimizer();

        // Act
        var checkpointId = manager.QueueSaveAsync(model, optimizer, new SaveOptions
        {
            CheckpointPrefix = "async_model"
        });

        // Assert
        Assert.NotNull(checkpointId);
        Assert.Equal(CheckpointTaskStatus.Queued, manager.GetTaskStatus(checkpointId));
        Assert.Equal(1, manager.QueueSize);
        Assert.Equal(1, manager.ActiveTaskCount);
    }

    [Fact]
    public async Task QueueSaveAsync_WhenQueueFull_ThrowsException()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 2);
        var model = new TestModel();
        var optimizer = new TestOptimizer();

        // Act
        manager.QueueSaveAsync(model, optimizer, new SaveOptions { CheckpointPrefix = "async_1" });
        manager.QueueSaveAsync(model, optimizer, new SaveOptions { CheckpointPrefix = "async_2" });

        // Assert
        Assert.Throws<CheckpointQueueFullException>(() =>
        {
            manager.QueueSaveAsync(model, optimizer, new SaveOptions { CheckpointPrefix = "async_3" });
        });
    }

    [Fact]
    public async Task WaitForCompletionAsync_WithValidCheckpoint_ReturnsResult()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5);
        var model = new TestModel();
        var optimizer = new TestOptimizer();

        var checkpointId = manager.QueueSaveAsync(model, optimizer, new SaveOptions
        {
            CheckpointPrefix = "async_model"
        });

        // Act
        var result = await manager.WaitForCompletionAsync(checkpointId, TimeSpan.FromSeconds(10));

        // Assert
        Assert.True(result.Success);
        Assert.NotNull(result.CheckpointPath);
        Assert.Equal(CheckpointTaskStatus.Completed, manager.GetTaskStatus(checkpointId));
    }

    [Fact]
    public async Task WaitForCompletionAsync_WithInvalidCheckpointId_ThrowsException()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            manager.WaitForCompletionAsync("invalid_id", TimeSpan.FromSeconds(1)));
    }

    [Fact]
    public async Task CancelCheckpoint_WithQueuedTask_CancelsSuccessfully()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5);
        var model = new TestModel();
        var optimizer = new TestOptimizer();

        var checkpointId = manager.QueueSaveAsync(model, optimizer, new SaveOptions
        {
            CheckpointPrefix = "async_model"
        });

        // Act
        var cancelled = manager.CancelCheckpoint(checkpointId);

        // Assert
        Assert.True(cancelled);
        Assert.Equal(CheckpointTaskStatus.Cancelled, manager.GetTaskStatus(checkpointId));
    }

    [Fact]
    public async Task CancelCheckpoint_WithRunningTask_DoesNotCancel()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 1);
        var model = new TestModel();
        var optimizer = new TestOptimizer();

        var checkpointId = manager.QueueSaveAsync(model, optimizer, new SaveOptions
        {
            CheckpointPrefix = "async_model"
        });

        // Wait for task to start running
        await Task.Delay(100);

        // Act
        var cancelled = manager.CancelCheckpoint(checkpointId);

        // Assert
        Assert.False(cancelled);
        Assert.NotEqual(CheckpointTaskStatus.Cancelled, manager.GetTaskStatus(checkpointId));
    }

    [Fact]
    public async Task GetTaskStatus_WithValidCheckpointId_ReturnsStatus()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5);
        var model = new TestModel();
        var optimizer = new TestOptimizer();

        var checkpointId = manager.QueueSaveAsync(model, optimizer, new SaveOptions
        {
            CheckpointPrefix = "async_model"
        });

        // Act
        var status = manager.GetTaskStatus(checkpointId);

        // Assert
        Assert.NotNull(status);
        Assert.Equal(CheckpointTaskStatus.Queued, status);
    }

    [Fact]
    public async Task GetTaskStatus_WithInvalidCheckpointId_ReturnsNull()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5);

        // Act
        var status = manager.GetTaskStatus("invalid_id");

        // Assert
        Assert.Null(status);
    }

    [Fact]
    public async Task GetActiveTasks_WithMultipleTasks_ReturnsAllActiveTasks()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5);
        var model = new TestModel();
        var optimizer = new TestOptimizer();

        var id1 = manager.QueueSaveAsync(model, optimizer, new SaveOptions { CheckpointPrefix = "async_1" });
        var id2 = manager.QueueSaveAsync(model, optimizer, new SaveOptions { CheckpointPrefix = "async_2" });
        var id3 = manager.QueueSaveAsync(model, optimizer, new SaveOptions { CheckpointPrefix = "async_3" });

        // Act
        var activeTasks = manager.GetActiveTasks();

        // Assert
        Assert.Equal(3, activeTasks.Count);
        Assert.Contains(activeTasks, t => t.Id == id1);
        Assert.Contains(activeTasks, t => t.Id == id2);
        Assert.Contains(activeTasks, t => t.Id == id3);
    }

    [Fact]
    public async Task GetTaskInfo_WithValidCheckpointId_ReturnsTaskInfo()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5);
        var model = new TestModel();
        var optimizer = new TestOptimizer();

        var checkpointId = manager.QueueSaveAsync(model, optimizer, new SaveOptions
        {
            CheckpointPrefix = "async_model"
        });

        // Act
        var taskInfo = manager.GetTaskInfo(checkpointId);

        // Assert
        Assert.NotNull(taskInfo);
        Assert.Equal(checkpointId, taskInfo.Id);
        Assert.Equal(CheckpointTaskType.Save, taskInfo.Type);
        Assert.Equal(CheckpointTaskStatus.Queued, taskInfo.Status);
        Assert.True(taskInfo.TimeSinceQueued.TotalMilliseconds >= 0);
    }

    [Fact]
    public async Task GetTaskInfo_WithInvalidCheckpointId_ReturnsNull()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5);

        // Act
        var taskInfo = manager.GetTaskInfo("invalid_id");

        // Assert
        Assert.Null(taskInfo);
    }

    [Fact]
    public async Task ConcurrentQueues_ProcessesAllTasks()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        using var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 10);
        var model = new TestModel();
        var optimizer = new TestOptimizer();

        var tasks = new List<Task<(string id, CheckpointTaskResult result)>>();

        // Act
        for (int i = 0; i < 5; i++)
        {
            var checkpointId = manager.QueueSaveAsync(model, optimizer, new SaveOptions
            {
                CheckpointPrefix = $"async_{i}"
            });

            tasks.Add(Task.Run(async () =>
            {
                var result = await manager.WaitForCompletionAsync(checkpointId, TimeSpan.FromSeconds(10));
                return (checkpointId, result);
            }));
        }

        var results = await Task.WhenAll(tasks);

        // Assert
        Assert.All(results, r => Assert.True(r.result.Success));
    }

    [Fact]
    public async Task Dispose_ShutsDownWorkerGracefully()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);
        var model = new TestModel();
        var optimizer = new TestOptimizer();

        // Act
        using (var manager = new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize: 5))
        {
            manager.QueueSaveAsync(model, optimizer, new SaveOptions { CheckpointPrefix = "async_1" });
            // Worker should shut down when disposed
        }

        // Assert - no exception thrown during dispose
    }

    [Fact]
    public async Task CheckpointTaskInfo_Duration_CalculatesCorrectly()
    {
        // Arrange
        var startedAt = DateTime.UtcNow.AddMilliseconds(-100);
        var completedAt = DateTime.UtcNow;
        var taskInfo = new CheckpointTaskInfo
        {
            Id = "test_id",
            Type = CheckpointTaskType.Save,
            Status = CheckpointTaskStatus.Completed,
            QueuedAt = startedAt,
            StartedAt = startedAt,
            CompletedAt = completedAt
        };

        // Act
        var duration = taskInfo.Duration;

        // Assert
        Assert.NotNull(duration);
        Assert.True(duration.Value.TotalMilliseconds >= 90 && duration.Value.TotalMilliseconds <= 150);
    }

    [Fact]
    public async Task CheckpointTaskInfo_IsActive_ReturnsCorrectStatus()
    {
        // Arrange
        var queuedTask = new CheckpointTaskInfo
        {
            Id = "queued_id",
            Type = CheckpointTaskType.Save,
            Status = CheckpointTaskStatus.Queued,
            QueuedAt = DateTime.UtcNow
        };

        var runningTask = new CheckpointTaskInfo
        {
            Id = "running_id",
            Type = CheckpointTaskType.Save,
            Status = CheckpointTaskStatus.Running,
            QueuedAt = DateTime.UtcNow
        };

        var completedTask = new CheckpointTaskInfo
        {
            Id = "completed_id",
            Type = CheckpointTaskType.Save,
            Status = CheckpointTaskStatus.Completed,
            QueuedAt = DateTime.UtcNow
        };

        // Assert
        Assert.True(queuedTask.IsActive);
        Assert.True(runningTask.IsActive);
        Assert.False(completedTask.IsActive);
    }

    [Fact]
    public async Task CheckpointTaskResult_CreateSuccess_ReturnsSuccessfulResult()
    {
        // Act
        var result = CheckpointTaskResult.CreateSuccess("/path/to/checkpoint", TimeSpan.FromSeconds(1), 1024);

        // Assert
        Assert.True(result.Success);
        Assert.Equal("/path/to/checkpoint", result.CheckpointPath);
        Assert.Equal(TimeSpan.FromSeconds(1), result.Duration);
        Assert.Equal(1024, result.BytesWritten);
        Assert.Null(result.Error);
    }

    [Fact]
    public async Task CheckpointTaskResult_CreateFailure_ReturnsFailedResult()
    {
        // Act
        var exception = new Exception("Test exception");
        var result = CheckpointTaskResult.CreateFailure("Test error", exception);

        // Assert
        Assert.False(result.Success);
        Assert.Equal("Test error", result.Error);
        Assert.Equal(exception, result.Exception);
        Assert.Null(result.CheckpointPath);
    }

    [Fact]
    public async Task CreateAsyncManager_ExtensionMethod_CreatesManager()
    {
        // Arrange
        var coordinator = new MockCoordinator();
        var storage = new LocalFileSystemStorage(_testCheckpointDir);
        var checkpoint = new DistributedCheckpoint(coordinator, storage);

        // Act
        using var manager = checkpoint.CreateAsyncManager(maxQueueSize: 5);

        // Assert
        Assert.NotNull(manager);
        Assert.Equal(5, manager.MaxQueueSize);
    }
}
