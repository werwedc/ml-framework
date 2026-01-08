# Spec: Point-to-Point Communication

## Overview
Implement point-to-point communication operations (Send/Receive) for direct communication between specific ranks.

## Dependencies
- `spec_communication_interfaces.md`

## Technical Requirements

### 1. Point-to-Point Interface
Define interface for point-to-point communication.

```csharp
namespace MLFramework.Communication.PointToPoint
{
    /// <summary>
    /// Interface for point-to-point communication operations
    /// </summary>
    public interface IPointToPointCommunication : ICommunicationBackend
    {
        /// <summary>
        /// Send tensor to a specific rank
        /// </summary>
        /// <param name="tensor">Tensor to send</param>
        /// <param name="destinationRank">Rank to send to</param>
        /// <param name="tag">Message tag for matching (optional)</param>
        void Send<T>(Tensor<T> tensor, int destinationRank, int tag = 0);

        /// <summary>
        /// Receive tensor from a specific rank
        /// </summary>
        /// <param name="destinationRank">Rank to receive from</param>
        /// <param name="tag">Message tag for matching (optional)</param>
        /// <returns>Received tensor</returns>
        Tensor<T> Receive<T>(int sourceRank, int tag = 0);

        /// <summary>
        /// Receive tensor with known shape (more efficient)
        /// </summary>
        Tensor<T> Receive<T>(int sourceRank, Tensor<T> template, int tag = 0);

        /// <summary>
        /// Non-blocking send
        /// </summary>
        ICommunicationHandle SendAsync<T>(Tensor<T> tensor, int destinationRank, int tag = 0);

        /// <summary>
        /// Non-blocking receive
        /// </summary>
        ICommunicationHandle ReceiveAsync<T>(int sourceRank, int tag = 0);

        /// <summary>
        /// Non-blocking receive with known shape
        /// </summary>
        ICommunicationHandle ReceiveAsync<T>(int sourceRank, Tensor<T> template, int tag = 0);

        /// <summary>
        /// Probe for incoming message
        /// </summary>
        /// <param name="sourceRank">Rank to probe</param>
        /// <param name="tag">Message tag (use -1 for any tag)</param>
        /// <returns>Message info or null if no message</returns>
        MessageInfo? Probe(int sourceRank, int tag = 0);
    }

    /// <summary>
    /// Information about an incoming message
    /// </summary>
    public class MessageInfo
    {
        public int SourceRank { get; set; }
        public int Tag { get; set; }
        public long Count { get; set; } // Number of elements
        public Type DataType { get; set; }
    }
}
```

### 2. Send Operation
Synchronous send implementation.

```csharp
namespace MLFramework.Communication.PointToPoint
{
    /// <summary>
    /// Synchronous send operation
    /// </summary>
    public static class Send
    {
        /// <summary>
        /// Send tensor to a specific rank
        /// </summary>
        public static void SendTensor<T>(
            IPointToPointCommunication backend,
            Tensor<T> tensor,
            int destinationRank,
            int tag = 0)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (destinationRank < 0 || destinationRank >= backend.WorldSize)
                throw new ArgumentOutOfRangeException(nameof(destinationRank));

            if (destinationRank == backend.Rank)
                throw new ArgumentException("Cannot send to self");

            if (tag < 0)
                throw new ArgumentOutOfRangeException(nameof(tag));

            backend.Send(tensor, destinationRank, tag);
        }

        /// <summary>
        /// Send multiple tensors
        /// </summary>
        public static void SendTensors<T>(
            IPointToPointCommunication backend,
            IEnumerable<Tensor<T>> tensors,
            int destinationRank,
            int tag = 0)
        {
            foreach (var tensor in tensors)
            {
                SendTensor(backend, tensor, destinationRank, tag++);
            }
        }
    }
}
```

### 3. Receive Operation
Synchronous receive implementation.

```csharp
namespace MLFramework.Communication.PointToPoint
{
    /// <summary>
    /// Synchronous receive operation
    /// </summary>
    public static class Receive
    {
        /// <summary>
        /// Receive tensor from a specific rank
        /// </summary>
        public static Tensor<T> ReceiveTensor<T>(
            IPointToPointCommunication backend,
            int sourceRank,
            int tag = 0)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (sourceRank < 0 || sourceRank >= backend.WorldSize)
                throw new ArgumentOutOfRangeException(nameof(sourceRank));

            if (sourceRank == backend.Rank)
                throw new ArgumentException("Cannot receive from self");

            if (tag < 0)
                throw new ArgumentOutOfRangeException(nameof(tag));

            return backend.Receive<T>(sourceRank, tag);
        }

        /// <summary>
        /// Receive tensor with known shape (more efficient)
        /// </summary>
        public static Tensor<T> ReceiveTensor<T>(
            IPointToPointCommunication backend,
            int sourceRank,
            Tensor<T> template,
            int tag = 0)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (template == null)
                throw new ArgumentNullException(nameof(template));

            if (sourceRank < 0 || sourceRank >= backend.WorldSize)
                throw new ArgumentOutOfRangeException(nameof(sourceRank));

            if (sourceRank == backend.Rank)
                throw new ArgumentException("Cannot receive from self");

            if (tag < 0)
                throw new ArgumentOutOfRangeException(nameof(tag));

            return backend.Receive(sourceRank, template, tag);
        }

        /// <summary>
        /// Receive multiple tensors
        /// </summary>
        public static List<Tensor<T>> ReceiveTensors<T>(
            IPointToPointCommunication backend,
            int sourceRank,
            int count,
            int tag = 0)
        {
            var result = new List<Tensor<T>>();
            for (int i = 0; i < count; i++)
            {
                result.Add(ReceiveTensor(backend, sourceRank, tag + i));
            }
            return result;
        }
    }
}
```

### 4. Send-Receive Pair
Implement common send-receive patterns.

```csharp
namespace MLFramework.Communication.PointToPoint
{
    /// <summary>
    /// Common send-receive patterns
    /// </summary>
    public static class SendReceivePair
    {
        /// <summary>
        /// Send to one rank and receive from another (non-blocking)
        /// </summary>
        public static (ICommunicationHandle sendHandle, ICommunicationHandle receiveHandle) SendReceiveAsync<T>(
            IPointToPointCommunication backend,
            Tensor<T> sendTensor,
            int destinationRank,
            int sourceRank,
            int sendTag = 0,
            int receiveTag = 0)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (sendTensor == null)
                throw new ArgumentNullException(nameof(sendTensor));

            if (destinationRank == sourceRank)
                throw new ArgumentException("Destination and source ranks must be different");

            // Post receive first to avoid deadlock
            var receiveHandle = backend.ReceiveAsync<T>(sourceRank, receiveTag);
            var sendHandle = backend.SendAsync(sendTensor, destinationRank, sendTag);

            return (sendHandle, receiveHandle);
        }

        /// <summary>
        /// Send to one rank and receive from another (blocking)
        /// </summary>
        public static Tensor<T> SendReceive<T>(
            IPointToPointCommunication backend,
            Tensor<T> sendTensor,
            int destinationRank,
            int sourceRank,
            int sendTag = 0,
            int receiveTag = 0)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (sendTensor == null)
                throw new ArgumentNullException(nameof(sendTensor));

            if (destinationRank == sourceRank)
                throw new ArgumentException("Destination and source ranks must be different");

            // Use async to avoid deadlock
            var (sendHandle, receiveHandle) = SendReceiveAsync(
                backend, sendTensor, destinationRank, sourceRank, sendTag, receiveTag);

            // Wait for both operations
            sendHandle.Wait();
            receiveHandle.Wait();

            return receiveHandle.GetResult<T>();
        }

        /// <summary>
        /// Ring send-receive pattern (each rank sends to next, receives from previous)
        /// </summary>
        public static Tensor<T> RingSendReceive<T>(
            IPointToPointCommunication backend,
            Tensor<T> sendTensor,
            int tag = 0)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (sendTensor == null)
                throw new ArgumentNullException(nameof(sendTensor));

            int nextRank = (backend.Rank + 1) % backend.WorldSize;
            int prevRank = (backend.Rank - 1 + backend.WorldSize) % backend.WorldSize;

            return SendReceive(backend, sendTensor, nextRank, prevRank, tag, tag);
        }
    }
}
```

### 5. Asynchronous Send/Receive
Non-blocking point-to-point operations.

```csharp
namespace MLFramework.Communication.PointToPoint.Async
{
    /// <summary>
    /// Asynchronous send operation
    /// </summary>
    public static class SendAsync
    {
        public static ICommunicationHandle SendTensorAsync<T>(
            IPointToPointCommunication backend,
            Tensor<T> tensor,
            int destinationRank,
            int tag = 0)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            if (destinationRank < 0 || destinationRank >= backend.WorldSize)
                throw new ArgumentOutOfRangeException(nameof(destinationRank));

            if (destinationRank == backend.Rank)
                throw new ArgumentException("Cannot send to self");

            return backend.SendAsync(tensor, destinationRank, tag);
        }
    }

    /// <summary>
    /// Asynchronous receive operation
    /// </summary>
    public static class ReceiveAsync
    {
        public static ICommunicationHandle ReceiveTensorAsync<T>(
            IPointToPointCommunication backend,
            int sourceRank,
            int tag = 0)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (sourceRank < 0 || sourceRank >= backend.WorldSize)
                throw new ArgumentOutOfRangeException(nameof(sourceRank));

            if (sourceRank == backend.Rank)
                throw new ArgumentException("Cannot receive from self");

            return backend.ReceiveAsync<T>(sourceRank, tag);
        }

        public static ICommunicationHandle ReceiveTensorAsync<T>(
            IPointToPointCommunication backend,
            int sourceRank,
            Tensor<T> template,
            int tag = 0)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (template == null)
                throw new ArgumentNullException(nameof(template));

            if (sourceRank < 0 || sourceRank >= backend.WorldSize)
                throw new ArgumentOutOfRangeException(nameof(sourceRank));

            if (sourceRank == backend.Rank)
                throw new ArgumentException("Cannot receive from self");

            return backend.ReceiveAsync(sourceRank, template, tag);
        }
    }
}
```

### 6. Probe Operation
Check for incoming messages without blocking.

```csharp
namespace MLFramework.Communication.PointToPoint
{
    /// <summary>
    /// Probe for incoming messages
    /// </summary>
    public static class Probe
    {
        /// <summary>
        /// Check if there's an incoming message from a specific rank
        /// </summary>
        /// <returns>Message info or null if no message available</returns>
        public static MessageInfo? ProbeMessage(
            IPointToPointCommunication backend,
            int sourceRank,
            int tag = 0)
        {
            if (backend == null)
                throw new ArgumentNullException(nameof(backend));

            if (sourceRank < 0 || sourceRank >= backend.WorldSize)
                throw new ArgumentOutOfRangeException(nameof(sourceRank));

            return backend.Probe(sourceRank, tag);
        }

        /// <summary>
        /// Wait for message with timeout
        /// </summary>
        /// <returns>Message info or null if timeout</returns>
        public static MessageInfo? WaitForMessage(
            IPointToPointCommunication backend,
            int sourceRank,
            int timeoutMs = 5000,
            int tag = 0)
        {
            var stopwatch = Stopwatch.StartNew();
            while (stopwatch.ElapsedMilliseconds < timeoutMs)
            {
                var info = ProbeMessage(backend, sourceRank, tag);
                if (info != null)
                {
                    return info;
                }
                Thread.Sleep(10);
            }
            return null;
        }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/PointToPoint/IPointToPointCommunication.cs`
   - `src/MLFramework/Communication/PointToPoint/MessageInfo.cs`
   - `src/MLFramework/Communication/PointToPoint/Send.cs`
   - `src/MLFramework/Communication/PointToPoint/Receive.cs`
   - `src/MLFramework/Communication/PointToPoint/SendReceivePair.cs`
   - `src/MLFramework/Communication/PointToPoint/Async/SendAsync.cs`
   - `src/MLFramework/Communication/PointToPoint/Async/ReceiveAsync.cs`
   - `src/MLFramework/Communication/PointToPoint/Probe.cs`

2. **Design Decisions:**
   - Tags allow multiple simultaneous messages between same ranks
   - Send-receive pairs help avoid deadlocks
   - Ring pattern is common for pipeline parallelism
   - Probe enables efficient message handling

3. **Error Handling:**
   - Validate ranks are within valid range
   - Prevent sending/receiving to self
   - Handle timeout scenarios
   - Clear error messages for debugging

4. **Performance Considerations:**
   - Non-blocking operations enable compute-communication overlap
   - Probe helps avoid blocking on receives
   - Known shape receives are more efficient

## Testing Requirements
- Unit tests for send/receive pairs
- Tests for deadlocks (should not occur)
- Tests for send-receive patterns
- Tests for probe operations
- Integration tests with multiple ranks

## Success Criteria
- All point-to-point operations compile and pass tests
- Send/receive correctly transfers data between ranks
- Non-blocking operations track completion state
- Ring pattern works correctly
- Probe detects incoming messages
