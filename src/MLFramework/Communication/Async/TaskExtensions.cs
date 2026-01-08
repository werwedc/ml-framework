namespace MLFramework.Communication.Async;

using System;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;

/// <summary>
/// Extension methods for CommunicationOperationQueue
/// </summary>
public static class CommunicationOperationQueueExtensions
{
    /// <summary>
    /// Get results from all completed operations in the queue
    /// </summary>
    public static List<Tensor> GetResults(this CommunicationOperationQueue queue)
    {
        // Note: This implementation needs access to queue internals
        // For now, returning an empty list as placeholder
        // A proper implementation would require queue to expose results
        
        // Alternative approach: maintain a separate list for results
        throw new NotImplementedException("GetResults requires access to queue internals. Consider modifying CommunicationOperationQueue to track results.");
    }
}
