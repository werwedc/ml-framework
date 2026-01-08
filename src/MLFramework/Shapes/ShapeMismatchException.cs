using System.Collections.ObjectModel;

namespace MLFramework.Shapes;

/// <summary>
/// Exception thrown when a shape mismatch is detected during static shape checking.
/// Provides detailed information about the operation, expected shapes, and actual shapes.
/// </summary>
public class ShapeMismatchException : Exception
{
    /// <summary>
    /// Gets the name of the operation that caused the shape mismatch.
    /// </summary>
    public string OperationName { get; }

    /// <summary>
    /// Gets the list of expected shapes for the operation.
    /// </summary>
    public ReadOnlyCollection<SymbolicShape> ExpectedShapes { get; }

    /// <summary>
    /// Gets the list of actual shapes that caused the mismatch.
    /// </summary>
    public ReadOnlyCollection<SymbolicShape> ActualShapes { get; }

    /// <summary>
    /// Gets additional details about the shape mismatch.
    /// </summary>
    public string Details { get; }

    /// <summary>
    /// Initializes a new instance of the ShapeMismatchException class.
    /// </summary>
    /// <param name="operationName">The name of the operation.</param>
    /// <param name="expectedShapes">The expected shapes.</param>
    /// <param name="actualShapes">The actual shapes.</param>
    /// <param name="details">Additional details about the mismatch.</param>
    /// <exception cref="ArgumentNullException">Thrown when operationName is null or empty.</exception>
    public ShapeMismatchException(
        string operationName,
        List<SymbolicShape> expectedShapes,
        List<SymbolicShape> actualShapes,
        string details = "")
        : base(GenerateMessage(operationName, expectedShapes, actualShapes, details))
    {
        OperationName = operationName ?? throw new ArgumentNullException(nameof(operationName));
        ExpectedShapes = new ReadOnlyCollection<SymbolicShape>(expectedShapes ?? new List<SymbolicShape>());
        ActualShapes = new ReadOnlyCollection<SymbolicShape>(actualShapes ?? new List<SymbolicShape>());
        Details = details ?? "";
    }

    /// <summary>
    /// Initializes a new instance of the ShapeMismatchException class with a custom message.
    /// </summary>
    /// <param name="operationName">The name of the operation.</param>
    /// <param name="expectedShapes">The expected shapes.</param>
    /// <param name="actualShapes">The actual shapes.</param>
    /// <param name="message">The custom message.</param>
    /// <param name="details">Additional details about the mismatch.</param>
    /// <exception cref="ArgumentNullException">Thrown when operationName is null or empty.</exception>
    public ShapeMismatchException(
        string operationName,
        List<SymbolicShape> expectedShapes,
        List<SymbolicShape> actualShapes,
        string message,
        string details)
        : base(message)
    {
        OperationName = operationName ?? throw new ArgumentNullException(nameof(operationName));
        ExpectedShapes = new ReadOnlyCollection<SymbolicShape>(expectedShapes ?? new List<SymbolicShape>());
        ActualShapes = new ReadOnlyCollection<SymbolicShape>(actualShapes ?? new List<SymbolicShape>());
        Details = details ?? "";
    }

    /// <summary>
    /// Initializes a new instance of the ShapeMismatchException class for serialization.
    /// </summary>
    /// <param name="info">Serialization info.</param>
    /// <param name="context">Streaming context.</param>
    protected ShapeMismatchException(
        System.Runtime.Serialization.SerializationInfo info,
        System.Runtime.Serialization.StreamingContext context)
        : base(info, context)
    {
        OperationName = info.GetString(nameof(OperationName)) ?? "";
        var expectedShapes = (List<SymbolicShape>?)info.GetValue(nameof(ExpectedShapes), typeof(List<SymbolicShape>));
        var actualShapes = (List<SymbolicShape>?)info.GetValue(nameof(ActualShapes), typeof(List<SymbolicShape>));
        ExpectedShapes = new ReadOnlyCollection<SymbolicShape>(expectedShapes ?? new List<SymbolicShape>());
        ActualShapes = new ReadOnlyCollection<SymbolicShape>(actualShapes ?? new List<SymbolicShape>());
        Details = info.GetString(nameof(Details)) ?? "";
    }

    /// <summary>
    /// Generates a formatted error message from the exception details.
    /// </summary>
    private static string GenerateMessage(
        string operationName,
        List<SymbolicShape> expectedShapes,
        List<SymbolicShape> actualShapes,
        string details)
    {
        var message = $"Shape mismatch detected in operation '{operationName}'.";

        if (expectedShapes.Count > 0 || actualShapes.Count > 0)
        {
            message += Environment.NewLine;

            if (expectedShapes.Count > 0)
            {
                message += $"  Expected shapes: [{string.Join(", ", expectedShapes.Select(s => s.ToString()))}]";
            }

            if (actualShapes.Count > 0)
            {
                message += Environment.NewLine;
                message += $"  Actual shapes:   [{string.Join(", ", actualShapes.Select(s => s.ToString()))}]";
            }
        }

        if (!string.IsNullOrWhiteSpace(details))
        {
            message += Environment.NewLine;
            message += $"  Details: {details}";
        }

        return message;
    }

    /// <summary>
    /// Returns a string representation of this exception.
    /// </summary>
    /// <returns>A formatted error message.</returns>
    public override string ToString()
    {
        return Message;
    }
}
