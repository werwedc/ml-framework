# Spec: Response Scatterer

## Overview
Implement response scattering logic to efficiently distribute batch processing results back to individual requests while maintaining order and handling failures gracefully.

## Technical Requirements

### Response Scatterer Class
```csharp
namespace MLFramework.Serving;

/// <summary>
/// Distributes batch processing results back to individual requests
/// </summary>
public class ResponseScatterer<TResponse>
{
    /// <summary>
    /// Scatter responses to the corresponding queued requests
    /// </summary>
    /// <param name="requests">List of queued requests that were processed</param>
    /// <param name="responses">List of responses from batch processing (null if error occurred)</param>
    /// <param name="exception">Exception if batch processing failed</param>
    public void Scatter(
        IReadOnlyList<QueuedRequest<TRequest>> requests,
        List<TResponse> responses,
        Exception exception = null)
    {
        if (requests == null)
            throw new ArgumentNullException(nameof(requests));

        if (exception != null)
        {
            // Batch-level failure: set exception for all requests
            ScatterException(requests, exception);
            return;
        }

        if (responses == null)
        {
            throw new ArgumentNullException(nameof(responses));
        }

        if (responses.Count != requests.Count)
        {
            throw new ArgumentException(
                $"Response count ({responses.Count}) does not match request count ({requests.Count})",
                nameof(responses));
        }

        // Scatter responses maintaining order
        for (int i = 0; i < requests.Count; i++)
        {
            var request = requests[i];
            var response = responses[i];

            request.ResponseSource.TrySetResult(response);
        }
    }

    private void ScatterException(
        IReadOnlyList<QueuedRequest<TRequest>> requests,
        Exception exception)
    {
        foreach (var request in requests)
        {
            request.ResponseSource.TrySetException(exception);
        }
    }

    /// <summary>
    /// Scatter responses with selective failures
    /// </summary>
    /// <param name="requests">List of queued requests</param>
    /// <param name="responses">List of responses (null entries indicate failure)</param>
    /// <param name="exceptions">List of exceptions for failed requests (null entries indicate success)</param>
    public void ScatterWithPartialFailures(
        IReadOnlyList<QueuedRequest<TRequest>> requests,
        List<TResponse> responses,
        List<Exception> exceptions)
    {
        if (requests == null)
            throw new ArgumentNullException(nameof(requests));

        if (responses == null)
            throw new ArgumentNullException(nameof(responses));

        if (exceptions == null)
            throw new ArgumentNullException(nameof(exceptions));

        var requestCount = requests.Count;

        if (responses.Count != requestCount || exceptions.Count != requestCount)
        {
            throw new ArgumentException(
                "Responses and exceptions arrays must match request count");
        }

        for (int i = 0; i < requestCount; i++)
        {
            var request = requests[i];
            var exception = exceptions[i];
            var response = responses[i];

            if (exception != null)
            {
                request.ResponseSource.TrySetException(exception);
            }
            else if (response != null)
            {
                request.ResponseSource.TrySetResult(response);
            }
            else
            {
                // Neither response nor exception - this is an error case
                request.ResponseSource.TrySetException(
                    new InvalidOperationException("No response or exception provided for request"));
            }
        }
    }
}
```

## File Location
- **Path:** `src/Serving/ResponseScatterer.cs`

## Dependencies
- `QueuedRequest<TRequest>` (from spec_request_queue.md)

## Key Design Decisions

1. **Order Preservation**: Responses are distributed in the same order as requests
2. **Batch-Level Failures**: If the entire batch fails, all requests get the exception
3. **Partial Failures**: Support for selective failures within a batch
4. **TaskCompletionSource**: Efficient async response delivery without blocking
5. **Validation**: Strict validation of array lengths to prevent index errors

## Success Criteria
- Responses are delivered to correct callers
- Order is preserved (first request gets first response)
- Batch-level exceptions are properly distributed
- Partial failures are handled correctly
- Validation catches mismatched array sizes
- TaskCompletionSource is properly set for all requests

## Testing Requirements
- Test successful scattering of multiple responses
- Test batch-level exception distribution
- Test partial failures (mix of success and failure)
- Test validation with mismatched array sizes
- Test order preservation
- Test null inputs validation
- Test scatter with null exception (success case)
- Test scatter with null responses and exception
- Test all requests get their TaskCompletionSource completed

## Notes
- This is the MVP implementation
- Future enhancements could include:
  - Retry logic for individual failed requests
  - Response compression for large payloads
  - Priority-based response ordering
