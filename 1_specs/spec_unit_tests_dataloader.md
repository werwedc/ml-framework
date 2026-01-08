# Spec: Unit Tests - DataLoader Core

## Overview
Comprehensive unit and integration tests for the main DataLoader<T> implementation.

## Test Structure
```
tests/
  Data/
    DataLoaderTests.cs
    DataLoaderIntegrationTests.cs
```

## Test Cases

### 1. Constructor Tests

**Basic Constructor:**
- `Constructor_ValidParameters_CreatesDataloader` - Normal case
- `Constructor_NullDataset_ThrowsArgumentNullException` - Null validation
- `Constructor_NullConfig_ThrowsArgumentNullException` - Null validation
- `Constructor_StoresConfig_ReturnsSameConfig` - Config stored correctly

**Constructor with Different Configs:**
- `Constructor_SingleWorker_CreatesCorrectly` - Single worker
- `Constructor_MultipleWorkers_CreatesCorrectly` - Multiple workers
- `Constructor_ShuffleTrue_ConfiguresShuffling` - Shuffling enabled
- `Constructor_ShuffleFalse_DisablesShuffling` - Shuffling disabled

### 2. Properties Tests

**IsRunning Property:**
- `IsRunning_InitiallyFalse` - Not running initially
- `IsRunning_AfterStart_ReturnsTrue` - Returns true after start
- `IsRunning_AfterStop_ReturnsFalse` - Returns false after stop

**Config Property:**
- `Config_ReturnsProvidedConfig` - Returns original config

**BatchCount Property:**
- `BatchCount_CalculatedCorrectly` - Matches dataset size / batch size
- `BatchCount_UnevenDivision_RoundedUp` - Rounded up for uneven division
- `BatchCount_EmptyDataset_ReturnsZero` - Zero for empty dataset

### 3. Start Tests

**Basic Start:**
- `Start_InitializesWorkers` - Workers created
- `Start_InitializesQueue` - Queue initialized
- `Start_InitializesPrefetching` - Prefetching started
- `Start_StartsBackgroundProduction` - Workers begin producing batches
- `Start_UpdatesIsRunningToTrue` - IsRunning set to true

**Start After Stop:**
- `Start_AfterStop_RestartsCorrectly` - Can restart after stop

**Start Already Running:**
- `Start_WhenAlreadyRunning_ThrowsInvalidOperationException` - Cannot start twice

**Start with Config:**
- `Start_RespectsNumWorkersInConfig` - Correct number of workers
- `Start_RespectsBatchSizeInConfig` - Correct batch size
- `Start_RespectsQueueSizeInConfig` - Correct queue size

### 4. Stop Tests

**Basic Stop:**
- `Stop_StopsWorkers` - Workers stopped
- `Stop_StopsPrefetching` - Prefetching stopped
- `Stop_MarksQueueComplete` - Queue marked complete
- `Stop_UpdatesIsRunningToFalse` - IsRunning set to false

**Stop Graceful:**
- `Stop_AllowsQueueToDrain` - Queue drains before complete

**Stop Before Start:**
- `Stop_BeforeStart_NoEffect` - Safe to stop before start
- `Stop_BeforeStart_DoesNotThrow` - No exception

**Stop Multiple Times:**
- `Stop_MultipleCalls_Idempotent` - Multiple calls safe

### 5. Reset Tests

**Basic Reset:**
- `Reset_StopsIfRunning` - Stops if running
- `Reset_ClearsInternalState` - State cleared
- `Reset_ResetsBatchIterator` - Iterator reset
- `Reset_AfterReset_CanStartAgain` - Can restart after reset

**Reset Before Start:**
- `Reset_BeforeStart_NoEffect` - Safe to reset before start

**Reset During Iteration:**
- `Reset_DuringIteration_RestartsFromBeginning` - Restarts iteration

### 6. Synchronous Iterator Tests

**GetEnumerator Basic:**
- `GetEnumerator_StartsAutomatically_IfNotStarted` - Auto-start
- `GetEnumerator_ReturnsBatchesInOrder` - Batches in correct order
- `GetEnumerator_ConsumesAllBatches` - All batches consumed

**GetEnumerator Blocking:**
- `GetEnumerator_BlocksUntilBatchAvailable` - Blocks when queue empty
- `GetEnumerator_ResumeWhenBatchAvailable` - Resumes when batch ready

**GetEnumerator Errors:**
- `GetEnumerator_BeforeStart_ThrowsInvalidOperationException` - Exception if not started
- `GetEnumerator_AfterDispose_ThrowsObjectDisposedException` - Exception if disposed

**GetEnumerator Multiple Iterations:**
- `GetEnumerator_MultipleIterations_WorkCorrectly` - Multiple passes work
- `GetEnumerator_MultipleIterations_AfterReset` - Works after reset

### 7. Async Iterator Tests

**GetAsyncEnumerator Basic:**
- `GetAsyncEnumerator_StartsAutomatically_IfNotStarted` - Auto-start
- `GetAsyncEnumerator_ReturnsBatchesInOrder` - Batches in correct order
- `GetAsyncEnumerator_ConsumesAllBatches` - All batches consumed
- `GetAsyncEnumerator_WaitForNextBatch_Awaitable` - Returns awaitable task

**GetAsyncEnumerator Cancellation:**
- `GetAsyncEnumerator_CancellationToken_Cancelled_ThrowsOperationCanceledException` - Cancel throws
- `GetAsyncEnumerator_CancellationToken_StopDuringIteration` - Cancel during iteration

**GetAsyncEnumerator Errors:**
- `GetAsyncEnumerator_AfterDispose_ThrowsObjectDisposedException` - Exception if disposed

### 8. Batch Assembly Tests

**Batch Creation:**
- `CreateBatch_CorrectAssemblesItems` - Items assembled correctly
- `CreateBatch_BatchSize_Respected` - Batch size respected
- `CreateBatch_FinalBatch_SmallerIfNeeded` - Final batch smaller if needed

**Index Generation:**
- `GenerateBatchIndices_CorrectPartitioning` - Indices partitioned correctly
- `GenerateBatchIndices_CorrectBatchSize` - Batch size correct
- `GenerateBatchIndices_FinalBatch_HandledCorrectly` - Final batch handled

### 9. Shuffling Tests

**Shuffling Enabled:**
- `ShufflingTrue_BatchesInDifferentOrders` - Different orders on multiple runs
- `ShufflingTrue_SeedProvided_ReproducibleOrder` - Seed provides reproducibility
- `ShufflingTrue_WithShuffleDisabled_DifferentOrders` - Order changes when shuffle toggled

**Shuffling Disabled:**
- `ShufflingFalse_BatchesInSameOrder` - Same order on multiple runs
- `ShufflingFalse_WithShuffleEnabled_DifferentOrders` - Order changes when shuffle enabled

**Shuffling Consistency:**
- `Shuffling_SameSeed_SameOrder` - Same seed produces same order
- `Shuffling_DifferentSeeds_DifferentOrders` - Different seeds produce different orders

### 10. Pinned Memory Tests

**Pinned Memory Enabled:**
- `PinMemoryTrue_UsesPinnedMemory` - Pinned memory used
- `PinMemoryTrue_BatchesUsePinnedMemory` - Batches use pinned memory

**Pinned Memory Disabled:**
- `PinMemoryFalse_DoesNotUsePinnedMemory` - Pinned memory not used
- `PinMemoryFalse_BatchesUseRegularMemory` - Batches use regular memory

### 11. Worker Pool Integration Tests

**Worker Execution:**
- `Workers_ProduceBatchesContinuously` - Continuous production
- `Workers_ProduceCorrectNumberOfBatches` - Correct count
- `Workers_ProduceBatchesInOrder` - Batches in order

**Worker Failure:**
- `WorkerFailure_DataLoaderContinues_ContinuePolicy` - Continues with error handling policy

**Worker Concurrency:**
- `MultipleWorkers_RunConcurrently` - Workers run in parallel
- `MultipleWorkers_AllEnqueueToQueue_ThreadSafe` - Thread-safe enqueue

### 12. Prefetching Integration Tests

**Prefetching Behavior:**
- `Prefetching_Started_InitialQueueFilled` - Queue filled on start
- `Prefetching_ConsumerConsumes_PrefetchRefills` - Refills when consumed
- `Prefetching_WithPrefetchCount_CorrectBehavior` - Prefetch count respected

**Prefetching Disabled:**
- `PrefetchingDisabled_BatchesProducedOnDemand` - On-demand production

### 13. Statistics Tests

**Statistics Tracking:**
- `Statistics_BatchesLoaded_IncrementsCorrectly` - Batches counted correctly
- `Statistics_TotalSamples_CalculatedCorrectly` - Total samples correct
- `Statistics_AverageBatchTime_CalculatedCorrectly` - Average time accurate
- `Statistics_Throughput_CalculatedCorrectly` - Throughput accurate

**Statistics Reset:**
- `Statistics_AfterReset_Cleared` - Statistics cleared on reset

**Statistics Integration:**
- `Statistics_WorkerStats_Propagated` - Worker stats propagated
- `Statistics_PrefetchStats_Propagated` - Prefetch stats propagated

### 14. Event Tests (Optional)

**WorkerError Event:**
- `WorkerError_RaisedWhenWorkerFails` - Event raised on worker failure

**RecoveryComplete Event:**
- `RecoveryComplete_RaisedAfterRecovery` - Event raised after recovery

**CriticalFailure Event:**
- `CriticalFailure_RaisedWhenAllWorkersFail` - Event raised on critical failure

### 15. Edge Cases

**Empty Dataset:**
- `EmptyDataset_BatchCountZero` - Batch count is zero
- `EmptyDataset_Iteration_NoBatches` - No batches yielded

**Single Item Dataset:**
- `SingleItemDataset_SingleBatch` - Single batch
- `SingleItemDataset_CorrectBatchSize` - Correct batch size

**Large Dataset:**
- `LargeDataset_CorrectBatchCount` - Correct count for large dataset
- `LargeDataset_AllBatchesProduced` - All batches produced

**Batch Size Larger Than Dataset:**
- `BatchSizeLargerThanDataset_SingleBatch` - Single batch

**Batch Size Equals Dataset:**
- `BatchSizeEqualsDataset_SingleBatch` - Single batch

**Batch Size One:**
- `BatchSizeOne_MultipleBatches` - One item per batch

**Zero Workers:**
- `ZeroWorkers_NoProduction` - No batches produced

### 16. Integration Tests

**End-to-End Flow:**
- `EndToEnd_DatasetToBatches_CompleteFlow` - Complete flow from dataset to batches
- `EndToEnd_IterationConsumesAllBatches` - All batches consumed in iteration
- `EndToEnd_MultipleIterations_WorkCorrectly` - Multiple iterations work

**Real Dataset:**
- `RealDataset_ImageDataLoader_WorkCorrectly` - Real dataset test (if available)

**Training Loop Simulation:**
- `TrainingLoopSimulation_CompleteTrainingCycle` - Simulate training loop

### 17. Dispose Tests

**Basic Dispose:**
- `Dispose_StopsIfRunning` - Stops if running
- `Dispose_CleansUpResources` - Resources cleaned up
- `Dispose_ReleasesPinnedMemory` - Pinned memory released
- `Dispose_CanCallMultipleTimes` - Safe to call multiple times

**Dispose During Iteration:**
- `Dispose_DuringIteration_StopsIteration` - Iteration stops

### 18. Performance Tests (Optional)

**Throughput:**
- `Throughput_BatchesPerSecond_Measured` - Measure batches/second
- `Throughput_SamplesPerSecond_Measured` - Measure samples/second

**Latency:**
- `Latency_FirstBatch_Measured` - Time to first batch
- `Latency_AverageBatch_Measured` - Average batch time

**Scalability:**
- `Scalability_IncreaseWorkers_ImprovesThroughput` - More workers improve throughput
- `Scalability_DatasetSize_LinearPerformance` - Performance scales with dataset size

## Test Utilities

**Helper Methods:**
```csharp
private IDataset<int> CreateTestDataset(int count)
{
    return new ArrayDataset<int>(Enumerable.Range(0, count).ToArray());
}

private DataLoader<int> CreateDataLoader(IDataset<int> dataset, DataLoaderConfig config)
{
    return new DataLoader<int>(dataset, config);
}

private DataLoaderConfig CreateTestConfig(
    int numWorkers = 2,
    int batchSize = 10,
    int prefetchCount = 1)
{
    return new DataLoaderConfig(
        numWorkers: numWorkers,
        batchSize: batchSize,
        prefetchCount: prefetchCount,
        queueSize: 10,
        shuffle: false,
        seed: 42,
        pinMemory: false);
}

private async Task<List<int>> ConsumeAllBatchesAsync(IDataLoader<int> dataloader)
{
    var batches = new List<int>();
    await foreach (var batch in dataloader)
    {
        batches.Add(batch);
    }
    return batches;
}
```

## Success Criteria
- [ ] All lifecycle methods tested (Start, Stop, Reset, Dispose)
- [ ] Synchronous iterator tested thoroughly
- [ ] Async iterator tested thoroughly
- [ ] Shuffling behavior verified
- [ ] Pinned memory behavior verified
- [ ] Worker pool integration tested
- [ ] Prefetching integration tested
- [ ] Statistics accuracy verified
- [ ] Events tested (if implemented)
- [ ] All edge cases covered
- [ ] Integration tests verify end-to-end flow
- [ ] Performance tests demonstrate improvements
- [ ] Test coverage > 95%
- [ ] All tests pass consistently

## Notes
- Use Task.WhenAll for concurrent operations
- Use CancellationTokenSource for cancellation tests
- Use timeout in tests to prevent deadlocks
- Performance tests should be marked with explicit attribute
- Integration tests may require mocking external dependencies
- Consider using Theory attribute for parameterized tests
- Tests should be deterministic (use fixed seeds for shuffling)
- Clean up resources in test teardown
