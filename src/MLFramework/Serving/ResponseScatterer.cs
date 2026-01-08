using System;
using System.Collections.Generic;

namespace MLFramework.Serving;

/// <summary>
/// Distributes batch processing results back to individual requests
/// </summary>
public class ResponseScatterer<TResponse>
{
    /// <summary>
    /// Scatter responses to the corresponding queued requests
    /// </summary>
    /// <typeparam name="TRequest">The request type</typeparam>
    /// <param name="requests">List of queued requests that were processed</param>
    /// <param name="responses">List of responses from batch processing (null if error occurred)</param>
    /// <param name="exception">Exception if batch processing failed</param>
    public void Scatter<TRequest>(
        IReadOnlyList<QueuedRequest<TRequest, TResponse>> requests,
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

    private void ScatterException<TRequest>(
        IReadOnlyList<QueuedRequest<TRequest, TResponse>> requests,
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
    /// <typeparam name="TRequest">The request type</typeparam>
    /// <param name="requests">List of queued requests</param>
    /// <param name="responses">List of responses (null entries indicate failure)</param>
    /// <param name="exceptions">List of exceptions for failed requests (null entries indicate success)</param>
    public void ScatterWithPartialFailures<TRequest>(
        IReadOnlyList<QueuedRequest<TRequest, TResponse>> requests,
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
