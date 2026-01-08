namespace MLFramework.Visualization.Graphs;

/// <summary>
/// Interface for logging computational graphs.
/// </summary>
public interface IGraphLogger
{
    /// <summary>
    /// Logs a computational graph.
    /// </summary>
    /// <param name="graph">The graph to log.</param>
    void LogGraph(ComputationalGraph graph);

    /// <summary>
    /// Logs a model by extracting its computational graph.
    /// </summary>
    /// <param name="model">The model to log.</param>
    void LogGraph(IModel model);

    /// <summary>
    /// Asynchronously logs a computational graph.
    /// </summary>
    /// <param name="graph">The graph to log.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    Task LogGraphAsync(ComputationalGraph graph);

    /// <summary>
    /// Asynchronously logs a model by extracting its computational graph.
    /// </summary>
    /// <param name="model">The model to log.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    Task LogGraphAsync(IModel model);

    /// <summary>
    /// Starts capturing a dynamic graph.
    /// </summary>
    /// <param name="name">Name for the captured graph.</param>
    void StartGraphCapture(string name);

    /// <summary>
    /// Stops capturing the dynamic graph.
    /// </summary>
    void StopGraphCapture();

    /// <summary>
    /// Gets the captured dynamic graph.
    /// </summary>
    /// <returns>The captured graph, or null if no capture is in progress.</returns>
    ComputationalGraph? GetCapturedGraph();

    /// <summary>
    /// Analyzes a graph and returns analysis results.
    /// </summary>
    /// <param name="graphId">ID of the graph to analyze.</param>
    /// <returns>Graph analysis results.</returns>
    GraphAnalysis AnalyzeGraph(string graphId);
}

/// <summary>
/// Simple interface for models that can be logged.
/// </summary>
public interface IModel
{
    /// <summary>
    /// Name of the model.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the computational graph for this model.
    /// </summary>
    /// <returns>The computational graph representing this model.</returns>
    ComputationalGraph GetGraph();
}
