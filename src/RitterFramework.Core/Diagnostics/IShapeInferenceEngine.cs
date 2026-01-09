namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Interface for inferring output tensor shapes for operations without execution.
/// </summary>
public interface IShapeInferenceEngine
{
    /// <summary>
    /// Infer output shape for an operation.
    /// </summary>
    long[] InferOutputShape(
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters = null);
}
