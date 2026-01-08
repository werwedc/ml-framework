# Spec: Unit Tests - Shared Queue

## Overview
Comprehensive unit tests for thread-safe shared queue implementation.

## Test Structure
```
tests/
  Data/
    SharedQueueTests.cs
```

## Test Cases

### 1. Constructor Tests

**Basic Constructor:**
- `Constructor_ValidCapacity_CreatesQueue` - Normal case
- `Constructor_CapacityOne_CreatesQueue` - Minimum capacity
- `Constructor_LargeCapacity_CreatesQueue` - Large queue (10,000+)

**Constructor with Cancellation Token:**
- `Constructor_WithCancellationToken_UsesToken` - Token stored correctly
- `Constructor_WithoutCancellationToken_NoToken` - Null token handled

**Invalid Constructor:**
- `Constructor_ZeroCapacity_ThrowsArgumentOutOfRangeException` - Zero capacity
- `Constructor_NegativeCapacity_ThrowsArgumentOutOfRangeException` - Negative capacity

### 2. Properties Tests

**Count Property:**
- `Count_EmptyQueue_ReturnsZero` - Empty queue has count 0
- `Count_AfterEnqueue_Increments` - Count increases after enqueue
- `Count_AfterDequeue_Decrements` - Count decreases after dequeue
- `Count_AfterCompleteAdding_Stable` - Count stable after complete

**IsCompleted Property:**
- `IsCompleted_Initial_ReturnsFalse` - Not completed initially
- `IsCompleted_AfterCompleteAdding_ReturnsTrue` - Returns true when complete
- `IsCompleted_AfterAllDequeued_ReturnsTrue` - Returns true when empty and complete

**Capacity Property:**
- `Capacity_ReturnsCorrectValue` - Matches constructor parameter

### 3. Enqueue Tests

**Basic Enqueue:**
- `Enqueue_SingleItem_IncreasesCount` - Count increments
- `Enqueue_MultipleItems_IncreasesCount` - Multiple enqueues
- `Enqueue_FullQueue_BlocksUntilSpace` - Blocking behavior
- `Enqueue_AfterCompleteAdding_ThrowsInvalidOperationException` - Cannot enqueue after complete

**Concurrent Enqueue:**
- `ConcurrentEnqueue_MultipleThreads_ThreadSafe` - Multiple producers
- `ConcurrentEnqueue_HighContention_RaceFree` - Stress test with 100 producers

### 4. Dequeue Tests

**Basic Dequeue:**
- `Dequeue_SingleItem_ReturnsCorrectItem` - FIFO order
- `Dequeue_MultipleItems_ReturnsInOrder` - Multiple items
- `Dequeue_EmptyQueue_BlocksUntilItem` - Blocking behavior
- `Dequeue_AfterCompleteAdding_ReturnsThenThrows` - Returns items then throws

**Concurrent Dequeue:**
- `ConcurrentDequeue_MultipleThreads_ThreadSafe` - Multiple consumers
- `ConcurrentDequeue_HighContention_RaceFree` - Stress test with 100 consumers

**Concurrent Enqueue/Dequeue:**
- `ConcurrentEnqueueDequeue_MultipleProducersConsumers_ThreadSafe` - Full producer-consumer
- `ConcurrentEnqueueDequeue_Balanced_Workload` - Equal producers and consumers
- `ConcurrentEnqueueDequeue_Unbalanced_Workload` - Different numbers of producers/consumers

### 5. TryEnqueue Tests

**TryEnqueue Success:**
- `TryEnqueue_SingleItem_ReturnsTrue` - Successful enqueue
- `TryEnqueue_NonFullQueue_ReturnsTrue` - Returns true when space available

**TryEnqueue Timeout:**
- `TryEnqueue_FullQueue_Timeout_ReturnsFalse` - Returns false on timeout
- `TryEnqueue_FullQueue_ShortTimeout_ReturnsFalse` - 10ms timeout
- `TryEnqueue_FullQueue_ZeroTimeout_ReturnsFalse` - 0ms timeout

### 6. TryDequeue Tests

**TryDequeue Success:**
- `TryDequeue_SingleItem_ReturnsTrue` - Successful dequeue
- `TryDequeue_NonEmptyQueue_ReturnsTrue` - Returns true when items available

**TryDequeue Timeout:**
- `TryDequeue_EmptyQueue_Timeout_ReturnsFalse` - Returns false on timeout
- `TryDequeue_EmptyQueue_ShortTimeout_ReturnsFalse` - 10ms timeout
- `TryDequeue_EmptyQueue_ZeroTimeout_ReturnsFalse` - 0ms timeout

### 7. TryPeek Tests

**TryPeek Success:**
- `TryPeek_SingleItem_ReturnsTrue` - Successful peek
- `TryPeek_NonEmptyQueue_ReturnsTrue` - Returns true when items available
- `TryPeek_DoesNotRemoveItem` - Item still in queue after peek
- `TryPeek_ReturnsNextItem` - Returns correct item (FIFO)

**TryPeek Failure:**
- `TryPeek_EmptyQueue_ReturnsFalse` - Returns false when empty
- `TryPeek_CompletedEmptyQueue_ReturnsFalse` - Returns false when complete and empty

### 8. CompleteAdding Tests

**Basic CompleteAdding:**
- `CompleteAdding_PreventsFurtherEnqueue` - Enqueue throws after complete
- `CompleteAdding_AllowsDequeueOfRemainingItems` - Can still dequeue
- `CompleteAdding_MultipleCalls_Idempotent` - Multiple calls safe

**CompleteAdding Behavior:**
- `CompleteAdding_EmptyQueue_SetsIsCompleted` - IsCompleted true
- `CompleteAdding_NonEmptyQueue_IsCompletedFalseUntilEmpty` - False until empty
- `CompleteAdding_AfterCompleteAdding_ThrowsInvalidOperationException` - Cannot enqueue

### 9. WaitForCompletion Tests

**Basic WaitForCompletion:**
- `WaitForCompletion_EmptyCompletedQueue_ReturnsImmediately` - No wait
- `WaitForCompletion_NonEmptyCompletedQueue_WaitsForEmpty` - Waits for all items
- `WaitForCompletion_WithoutCompleteAdding_ReturnsAfterItems` - Waits for items

**WaitForCompletion with Cancellation:**
- `WaitForCompletion_CancellationToken_ThrowsOperationCanceledException` - Cancel during wait
- `WaitForCompletion_AlreadyCompleted_ReturnsImmediately` - No wait if already complete

### 10. Shutdown Tests

**Basic Shutdown:**
- `Shutdown_ImmediatelyStopsQueue` - Queue stops immediately
- `Shutdown_AfterShutdown_CannotEnqueue` - Cannot enqueue after shutdown
- `Shutdown_AfterShutdown_CannotDequeue` - Cannot dequeue after shutdown
- `Shutdown_MultipleCalls_Idempotent` - Multiple calls safe

**Shutdown During Operations:**
- `Shutdown_DuringEnqueue_ThrowsOperationCanceledException` - Cancel during enqueue
- `Shutdown_DuringDequeue_ThrowsOperationCanceledException` - Cancel during dequeue
- `Shutdown_MultipleWaitingTasks_AllCancelled` - All waiting tasks cancelled

### 11. Cancellation Token Tests

**Token During Enqueue:**
- `CancelDuringEnqueue_ThrowsOperationCanceledException` - Cancel while blocking on enqueue

**Token During Dequeue:**
- `CancelDuringDequeue_ThrowsOperationCanceledException` - Cancel while blocking on dequeue

### 12. Statistics Tests

**Basic Statistics:**
- `Statistics_InitiallyZero_ReturnsZeroValues` - All statistics start at zero
- `Statistics_AfterEnqueue_IncrementsTotalEnqueued` - Enqueue count increases
- `Statistics_AfterDequeue_IncrementsTotalDequeued` - Dequeue count increases
- `Statistics_EnqueueDequeue_BothIncremented` - Both counters increment

**Queue Size Statistics:**
- `Statistics_MaxQueueSize_TrackedCorrectly` - Tracks maximum size
- `Statistics_AverageWaitTime_CalculatedCorrectly` - Average wait time accurate

### 13. Batch Operations Tests

**EnqueueBatch:**
- `EnqueueBatch_MultipleItems_IncrementsCountCorrectly` - Count increases by batch size
- `EnqueueBatch_FullQueue_BlocksUntilSpace` - Blocks until space for entire batch
- `EnqueueBatch_LargerThanCapacity_ThrowsOrBlocks` - Handle batch > capacity

**DequeueBatch:**
- `DequeueBatch_MultipleItems_ReturnsCorrectItems` - Returns correct items
- `DequeueBatch_RequestedMoreThanAvailable_ReturnsAvailable` - Returns available items
- `DequeueBatch_EmptyQueue_Timeout_ReturnsEmptyArray` - Returns empty array on timeout

### 14. Edge Cases

**Zero Capacity:**
- Should not be allowed (tested in constructor)

**Null Items:**
- `Enqueue_NullItem_Succeeds` - Null items allowed
- `Dequeue_NullItem_ReturnsNull` - Returns null correctly

**Full Queue:**
- `FullQueue_Enqueue_Blocks` - Blocks until space
- `FullQueue_TryEnquire_Timeout_ReturnsFalse` - Returns false on timeout

**Empty Queue:**
- `EmptyQueue_Dequeue_Blocks` - Blocks until item
- `EmptyQueue_TryDequeue_Timeout_ReturnsFalse` - Returns false on timeout

### 15. Performance Tests (Optional)

**Throughput:**
- `Throughput_SingleProducerConsumer_Measured` - Measure items/second
- `Throughput_MultipleProducersConsumers_Measured` - Measure with 10 producers/consumers

**Latency:**
- `Latency_EnqueueToDequeue_Measured` - Measure time from enqueue to dequeue

## Test Utilities

**Helper Methods:**
```csharp
private async Task EnqueueMultipleAsync(SharedQueue<int> queue, int count, int startValue = 0)
{
    for (int i = 0; i < count; i++)
    {
        queue.Enqueue(startValue + i);
    }
}

private async Task DequeueMultipleAsync(SharedQueue<int> queue, int count)
{
    for (int i = 0; i < count; i++)
    {
        queue.Dequeue();
    }
}

private async Task ProducerConsumerTest(int producers, int consumers, int itemsPerProducer)
{
    var queue = new SharedQueue<int>(itemsPerProducer * producers);
    var producerTasks = Enumerable.Range(0, producers)
        .Select(i => Task.Run(() => EnqueueMultipleAsync(queue, itemsPerProducer)));
    var consumerTasks = Enumerable.Range(0, consumers)
        .Select(i => Task.Run(() => DequeueMultipleAsync(queue, itemsPerProducer * producers / consumers)));
    await Task.WhenAll(producerTasks.Concat(consumerTasks));
}
```

## Success Criteria
- [ ] All blocking behaviors tested
- [ ] All timeout scenarios covered
- [ ] Cancellation token tested thoroughly
- [ ] Thread safety verified with concurrent tests
- [ ] Statistics accuracy verified
- [ ] Batch operations tested
- [ ] All edge cases covered
- [ ] Test coverage > 95%
- [ ] All tests pass consistently
- [ ] No race conditions in tests

## Notes
- Use Task.WhenAll for concurrent operations
- Use CancellationTokenSource for cancellation tests
- Use timeout in tests to prevent deadlocks
- Performance tests should be marked with explicit attribute
- Use TestCase for parameterized tests
- Consider using CollectionAssert for comparing collections
