using System.Text.Json;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Provides JSON serialization for shape mismatch exceptions.
/// </summary>
public static class JsonReportFormatter
{
    /// <summary>
    /// Converts a shape mismatch exception to a JSON string.
    /// </summary>
    /// <param name="exception">The exception to serialize.</param>
    /// <returns>A JSON string representation of the exception.</returns>
    public static string ToJson(ShapeMismatchException exception)
    {
        var data = new
        {
            layerName = exception.LayerName,
            operationType = exception.OperationType.ToString(),
            inputShapes = exception.InputShapes?.Select(s => new
            {
                dimensions = s,
                size = s != null ? s.Aggregate(1L, (a, b) => a * b) : 0L
            }),
            expectedShapes = exception.ExpectedShapes?.Select(s => new
            {
                dimensions = s
            }),
            problemDescription = exception.ProblemDescription,
            suggestedFixes = exception.SuggestedFixes,
            timestamp = System.DateTime.UtcNow.ToString("o"),
            stackTrace = exception.StackTrace
        };

        return JsonSerializer.Serialize(
            data,
            new JsonSerializerOptions { WriteIndented = true });
    }
}
