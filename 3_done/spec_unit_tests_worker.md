# Spec: Unit Tests - Worker Pool

## Overview
Comprehensive unit tests for worker pool management and parallel execution.

## Test Structure
```
tests/
  Data/
    WorkerPoolTests.cs
    WorkerContextTests.cs
```

## Test Cases

### 1. WorkerContext Tests

**Constructor Tests:**
- `Constructor_ValidParameters_CreatesContext` - Normal case
- `Constructor_EvenDivision_CalculatesCorrectRanges` - Equal partitions
- `Constructor_UnevenDivision_LastWorkerGetsRemainder` - Last worker gets extra items
- `Constructor_SingleItem_CorrectPartition` - Single item case
- `Constructor_EmptyDataset_ZeroLengthRange` - Empty dataset

**Property Tests:**
- `WorkerId_ReturnsCorrectValue` - Worker ID matches parameter
- `NumWorkers_ReturnsCorrectValue` - NumWorkers matches parameter
- `StartIndex_CalculatedCorrectly` - Start index correct
- `EndIndex_CalculatedCorrectly` - End index correct

**Edge Cases:**
- `WorkerContext_FirstWorker_StartsAtZero` - First worker starts at 0
- `WorkerContext_LastWorker_EndsAtTotal` - Last worker ends at total count
- `WorkerContext_MoreWorkersThanItems_CorrectPartition` - More workers than items

### 2. WorkerPool Constructor Tests

**Basic Constructor:**
- `Constructor_ValidParameters_CreatesPool` - Normal case
- `Constructor_ZeroWorkers_CreatesEmptyPool` - No workers
- `Constructor_MultipleWorkers_CreatesCorrectNumber` - Multiple workers

**Constructor with Cancellation Token:**
- `Constructor_WithCancellationToken_UsesToken` - Token stored correctly
- `Constructor_WithoutCancellationToken_NoToken` - Null token handled

**Invalid Constructor:**
- `Constructor_NullWorkerFunc_ThrowsArgumentNullException` - Null validation
- `Constructor_NullQueue_ThrowsArgumentNullException` - Null validation
- `Constructor_NegativeWorkers_ThrowsArgumentOutOfRangeException` - Negative validation

### 3. Properties Tests

**IsRunning Property:**
- `IsRunning_InitiallyFalse` - Not running initially
- `IsRunning_AfterStart_ReturnsTrue` - Returns true after start
- `IsRunning_AfterStop_ReturnsFalse` - Returns false after stop

**NumWorkers Property:**
- `NumWorkers_ReturnsCorrectValue` - Matches constructor parameter

**ActiveWorkers Property:**
- `ActiveWorkers_InitiallyZero` - No active workers initially
- `ActiveWorkers_AfterStart_EqualsNumWorkers` - All workers active after start
- `ActiveWorkers_WorkerCompletes_Decrements` - Decrements when worker completes

### 4. Start Tests

**Basic Start:**
- `Start_CreatesWorkerTasks` - Workers created and started
- `Start_UpdatesIsRunningToTrue` - IsRunning set to true
- `Start_UpdatesActiveWorkersToNumWorkers` - All workers active
- `Start_CalledMultipleTimes_ThrowsInvalidOperationException` - Cannot start twice

**Start with Worker Function:**
- `Start_WorkerFunctionCalledForEachWorker` - Each worker calls function
- `Start_WorkerFunctionReceivesCorrectWorkerId` - Worker IDs correct
- `Start_WorkerFunctionReceivesCancellationToken` - Token passed to workers

### 5. Stop Tests

**Basic Stop:**
- `Stop_StopsAllWorkers` - All workers stop
- `Stop_UpdatesIsRunningToFalse` - IsRunning set to false
- `Stop_UpdatesActiveWorkersToZero` - No active workers after stop
- `Stop_AfterStart_StopsGracefully` - Graceful shutdown

**Stop Timeout:**
- `Stop_WithTimeout_WaitsForWorkers` - Waits for workers to complete
- `Stop_TimeoutExceeded_ThrowsTimeoutException` - Timeout exception on timeout
- `Stop_ZeroTimeout_ReturnsImmediately` - Returns immediately with zero timeout

**Stop Before Start:**
- `Stop_BeforeStart_ReturnsSucceeds` - Safe to stop before start
- `Stop_MultipleCalls_Idempotent` - Multiple calls safe

**Stop with Cancellation:**
- `Stop_CancelsWorkerTasks` - Worker tasks cancelled
- `Stop_WorkersReceiveCancellationSignal` - Workers receive cancellation

### 6. WaitAsync Tests

**Basic WaitAsync:**
- `WaitAsync_AfterStart_WaitsForCompletion` - Waits for all workers
- `WaitAsync_AfterAllComplete_ReturnsImmediately` - No wait if complete

**WaitAsync with Errors:**
- `WaitAsync_WorkerThrows_ThrowsAggregateException` - Worker errors aggregated
- `WaitAsync_MultipleWorkersThrow_ContainsAllErrors` - All errors in aggregate

**WaitAsync Cancellation:**
- `WaitAsync_CancellationToken_ThrowsOperationCanceledException` - Cancel during wait

### 7. Worker Execution Tests

**Basic Worker Execution:**
- `WorkerExecution_ProducesItems_EnqueuesToQueue` - Items enqueued to queue
- `WorkerExecution_ProducesCorrectNumberOfItems` - Correct number of items
- `WorkerExecution_ProducesItemsInOrder_FIFO` - Items in correct order

**Worker with Cancellation:**
- `WorkerCancellation_StopsProduction` - Workers stop on cancellation
- `WorkerCancellation_EnqueuesNoMoreItems` - No more items after cancellation

**Worker with Exception:**
- `WorkerException_PropagatesToPool` - Exception propagated
- `WorkerException_OtherWorkersContinue_ContinuePolicy` - Other workers continue (if policy allows)

### 8. Concurrent Execution Tests

**Concurrent Worker Execution:**
- `ConcurrentWorkers_AllRunInParallel` - Workers run concurrently
- `ConcurrentWorkers_AllEnqueueToSameQueue_ThreadSafe` - Thread-safe enqueue
- `ConcurrentWorkers_ProduceDistinctItems_NoDuplicates` - Workers produce distinct items

**High Concurrency:**
- `HighConcurrency_100Workers_ThreadSafe` - 100 workers thread-safe
- `HighConcurrency_100Workers_ProducesCorrectCount` - Correct count produced

### 9. Worker Pool Factory Tests (Optional)

**CreateForDataset:**
- `CreateForDataset_CreatesWorkerPool` - Pool created correctly
- `CreateForDataset_WorkerCallsGetItem` - Workers call dataset.GetItem
- `CreateForDataset_PartitionsWorkCorrectly` - Work partitioned correctly

### 10. Event Hook Tests (Optional)

**WorkerStarted Event:**
- `WorkerStarted_RaisedForEachWorker` - Event raised for each worker
- `WorkerStarted_RaisedWithCorrectWorkerId` - Worker ID correct

**WorkerCompleted Event:**
- `WorkerCompleted_RaisedWhenWorkerFinishes` - Event raised on completion
- `WorkerCompleted_RaisedWithCorrectWorkerId` - Worker ID correct
- `WorkerCompleted_SuccessfulWorker_SuccessTrue` - Success true on success
- `WorkerCompleted_FailedWorker_SuccessFalse` - Success false on failure

### 11. Error Handling Tests

**Worker Throws Exception:**
- `WorkerThrowsException_PoolCapturesError` - Error captured
- `WorkerThrowsException_AggregateExceptionThrownInWaitAsync` - Aggregate in WaitAsync

**Crash During Execution:**
- `WorkerCrash_DetectedByPool` - Crash detected (if crash detection implemented)

**Graceful Degradation:**
- `SomeWorkersFail_OthersContinue_ContinuePolicy` - Continue with remaining workers

### 12. Performance Tests (Optional)

**Throughput:**
- `Throughput_SingleWorker_Measured` - Items/second with one worker
- `Throughput_MultipleWorkers_ScalesLinearly` - Throughput scales with workers

**Latency:**
- `Latency_FirstItem_Measured` - Time to first item
- `Latency_SubsequentItems_Measured` - Time to subsequent items

### 13. Edge Cases

**Zero Workers:**
- `ZeroWorkers_Start_NoWorkersCreated` - No workers created
- `ZeroWorkers_IsRunningFalse` - IsRunning remains false

**Single Worker:**
- `SingleWorker_CorrectlyExecutes` - Single worker works correctly

**Large Number of Workers:**
- `LargeNumberOfWorkers_100_CreatesCorrectly` - 100 workers created
- `LargeNumberOfWorkers_AllExecute_ThreadSafe` - All execute safely

**Empty Queue:**
- `WorkersEnqueueToEmptyQueue_Succeeds` - Empty queue handled

**Full Queue:**
- `WorkersEnqueueToFullQueue_Blocks` - Workers block on full queue

### 14. WorkerId Assignment Tests

**WorkerId Assignment:**
- `WorkerIds_AreZeroBased` - Workers IDs start at 0
- `WorkerIds_AreUnique` - Each worker has unique ID
- `WorkerIds_AreSequential` - Workers IDs are sequential

## Test Utilities

**Helper Methods:**
```csharp
private DataWorker<int> CreateSimpleWorker()
{
    return (workerId, token) => workerId * 100;
}

private DataWorker<int> CreateCountingWorker(SharedQueue<int> queue, int count)
{
    return (workerId, token) =>
    {
        for (int i = 0; i < count; i++)
        {
            queue.Enqueue(workerId * count + i);
        }
        return 0;
    };
}

private async Task EnqueueItemsAsync(WorkerPool<int> pool, SharedQueue<int> queue, int expectedCount)
{
    pool.Start();
    await pool.WaitAsync();
    Assert.AreEqual(expectedCount, queue.Count);
}
```

## Success Criteria
- [ ] All lifecycle methods tested (Start, Stop, WaitAsync)
- [ ] Worker execution behavior verified
- [ ] Concurrent execution tested thoroughly
- [ ] Cancellation token tested in all scenarios
- [ ] Error handling tested (worker exceptions)
- [ ] Event hooks tested (if implemented)
- [ ] Edge cases covered (zero workers, single worker, large number of workers)
- [ ] Worker context partitioning verified
- [ ] Factory methods tested (if implemented)
- [ ] Performance tests demonstrate scalability
- [ ] Test coverage > 95%
- [ ] All tests pass consistently

## Notes
- Use Task.WhenAll for concurrent operations
- Use CancellationTokenSource for cancellation tests
- Use timeout in tests to prevent deadlocks
- Performance tests should be marked with explicit attribute
- Use TestCase for parameterized tests
- Test with both normal and edge case worker counts
- Verify thread safety with high contention tests
