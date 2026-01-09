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

    /// <summary>
    /// Infer all intermediate shapes in a computation graph.
    /// </summary>
    /// <param name="graph">The computation graph.</param>
    /// <param name="inputShapes">Input shapes for graph inputs (node name -> shape).</param>
    /// <returns>Dictionary mapping node names to their output shapes.</returns>
    IDictionary<string, long[]> InferGraphShapes(
        ComputationGraph graph,
        IDictionary<string, long[]> inputShapes);

    /// <summary>
    /// Validate that shapes are compatible with operation.
    /// </summary>
    /// <param name="operationType">The operation type.</param>
    /// <param name="inputShapes">Input shapes.</param>
    /// <param name="operationParameters">Operation parameters.</param>
    /// <param name="errorMessage">Output error message if validation fails.</param>
    /// <returns>True if shapes are valid, false otherwise.</returns>
    bool ValidateOperation(
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        out string errorMessage);
}
