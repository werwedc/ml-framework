namespace MLFramework.Communication.Async;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed.Communication;

/// <summary>
/// Wrapper for async operations with error handling
/// </summary>
public static class AsyncOperationWrapper
{
    /// <summary>
    /// Execute async operation with timeout and error handling
    /// </summary>
    public static async Task<Tensor> ExecuteAsync(
        Func<Task<Tensor>> operation,
        int timeoutMs,
        CancellationToken cancellationToken = default)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(timeoutMs);

        try
        {
            return await operation().WithCancellation(cts.Token);
        }
        catch (OperationCanceledException) when (cts.Token.IsCancellationRequested)
        {
            throw new CommunicationTimeoutException($"Operation timed out after {timeoutMs}ms", TimeSpan.FromMilliseconds(timeoutMs));
        }
        catch (Exception ex)
        {
            throw new CommunicationException("Async operation failed", ex);
        }
    }

    /// <summary>
    /// Execute multiple async operations in parallel
    /// </summary>
    public static async Task<List<Tensor>> ExecuteAllAsync(
        IEnumerable<Func<Task<Tensor>>> operations,
        int timeoutMs = -1,
        CancellationToken cancellationToken = default)
    {
        var tasks = operations.Select(op => op()).ToList();

        if (timeoutMs > 0)
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            cts.CancelAfter(timeoutMs);

            try
            {
                await Task.WhenAll(tasks).WithCancellation(cts.Token);
            }
            catch (OperationCanceledException) when (cts.Token.IsCancellationRequested)
            {
                throw new CommunicationTimeoutException($"One or more operations timed out after {timeoutMs}ms", TimeSpan.FromMilliseconds(timeoutMs));
            }
        }
        else
        {
            await Task.WhenAll(tasks);
        }

        return tasks.Select(t => t.Result).ToList();
    }
}

/// <summary>
/// Extension methods for Task with cancellation support
/// </summary>
public static class TaskExtensions
{
    public static async Task<T> WithCancellation<T>(this Task<T> task, CancellationToken cancellationToken)
    {
        var tcs = new TaskCompletionSource<bool>();
        using (cancellationToken.Register(s => ((TaskCompletionSource<bool>)s!).TrySetResult(true), tcs))
        {
            if (task != await Task.WhenAny(task, tcs.Task))
            {
                throw new OperationCanceledException(cancellationToken);
            }
        }

        return await task;
    }
}
