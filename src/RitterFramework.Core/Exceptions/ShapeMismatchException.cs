namespace RitterFramework.Core;

/// <summary>
/// Custom exception class with rich diagnostic information for tensor shape mismatches.
/// </summary>
public class ShapeMismatchException : Exception
{
    /// <summary>
    /// Name of the layer where the mismatch occurred.
    /// </summary>
    public string? LayerName { get; set; }

    /// <summary>
    /// The type of operation that failed.
    /// </summary>
    public OperationType OperationType { get; set; }

    /// <summary>
    /// List of input shapes that caused the mismatch.
    /// </summary>
    public List<long[]> InputShapes { get; set; } = new();

    /// <summary>
    /// List of expected shapes for the operation.
    /// </summary>
    public List<long[]> ExpectedShapes { get; set; } = new();

    /// <summary>
    /// Human-readable description of the problem.
    /// </summary>
    public string? ProblemDescription { get; set; }

    /// <summary>
    /// List of suggested fixes for the mismatch.
    /// </summary>
    public List<string> SuggestedFixes { get; set; } = new();

    /// <summary>
    /// Default constructor for serialization.
    /// </summary>
    public ShapeMismatchException()
    {
    }

    /// <summary>
    /// Constructor with message.
    /// </summary>
    /// <param name="message">The error message.</param>
    public ShapeMismatchException(string message) : base(message)
    {
    }

    /// <summary>
    /// Constructor with message and inner exception.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    public ShapeMismatchException(string message, Exception innerException) : base(message, innerException)
    {
    }

    /// <summary>
    /// Full constructor with all diagnostic information.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="layerName">Name of the layer where the mismatch occurred.</param>
    /// <param name="operationType">The type of operation that failed.</param>
    /// <param name="inputShapes">List of input shapes that caused the mismatch.</param>
    /// <param name="expectedShapes">List of expected shapes for the operation.</param>
    /// <param name="problemDescription">Human-readable description of the problem.</param>
    /// <param name="suggestedFixes">List of suggested fixes for the mismatch.</param>
    public ShapeMismatchException(
        string message,
        string? layerName,
        OperationType operationType,
        List<long[]> inputShapes,
        List<long[]> expectedShapes,
        string? problemDescription = null,
        List<string>? suggestedFixes = null) : base(message)
    {
        LayerName = layerName;
        OperationType = operationType;
        InputShapes = inputShapes;
        ExpectedShapes = expectedShapes;
        ProblemDescription = problemDescription;
        SuggestedFixes = suggestedFixes ?? new List<string>();
    }

    /// <summary>
    /// Generates a formatted diagnostic report.
    /// </summary>
    /// <returns>A formatted string containing all diagnostic information.</returns>
    public string GetDiagnosticReport()
    {
        var report = new System.Text.StringBuilder();

        // Exception header
        report.AppendLine($"MLFramework.ShapeMismatchException: {Message}");

        // Add layer name if available
        if (!string.IsNullOrEmpty(LayerName))
        {
            report.AppendLine($"Layer: '{LayerName}'");
            report.AppendLine($"Operation: {OperationType}");
        }

        report.AppendLine();

        // Input shapes
        if (InputShapes.Count > 0)
        {
            for (int i = 0; i < InputShapes.Count; i++)
            {
                report.AppendLine($"Input shape {(InputShapes.Count > 1 ? $"[{i}] " : "")}:   [{string.Join(", ", InputShapes[i])}]");
            }
        }

        // Expected shapes
        if (ExpectedShapes.Count > 0)
        {
            for (int i = 0; i < ExpectedShapes.Count; i++)
            {
                report.AppendLine($"Expected {(ExpectedShapes.Count > 1 ? $"[{i}] " : "")}:      [{string.Join(", ", ExpectedShapes[i])}]");
            }
        }

        report.AppendLine();

        // Problem description
        if (!string.IsNullOrEmpty(ProblemDescription))
        {
            report.AppendLine($"Problem: {ProblemDescription}");
            report.AppendLine();
        }

        // Suggested fixes
        if (SuggestedFixes.Count > 0)
        {
            report.AppendLine("Suggested fixes:");
            for (int i = 0; i < SuggestedFixes.Count; i++)
            {
                report.AppendLine($"{i + 1}. {SuggestedFixes[i]}");
            }
        }

        return report.ToString();
    }
}
