using System.Collections.Concurrent;
using RitterFramework.Core;

namespace MLFramework.Diagnostics;

/// <summary>
/// Registry for diagnostic formatters that provides centralized access to
/// operation-specific error formatting and suggestion generation.
/// </summary>
public class DiagnosticFormatterRegistry
{
    private static readonly Lazy<DiagnosticFormatterRegistry> _instance =
        new Lazy<DiagnosticFormatterRegistry>(() => new DiagnosticFormatterRegistry());

    private readonly ConcurrentDictionary<OperationType, IDiagnosticFormatter> _formatters;

    /// <summary>
    /// Gets the singleton instance of the registry.
    /// </summary>
    public static DiagnosticFormatterRegistry Instance => _instance.Value;

    /// <summary>
    /// Private constructor for the singleton pattern.
    /// </summary>
    private DiagnosticFormatterRegistry()
    {
        _formatters = new ConcurrentDictionary<OperationType, IDiagnosticFormatter>();
    }

    /// <summary>
    /// Registers a diagnostic formatter for a specific operation type.
    /// </summary>
    /// <param name="formatter">The formatter to register.</param>
    public void Register(IDiagnosticFormatter formatter)
    {
        if (formatter == null)
        {
            throw new ArgumentNullException(nameof(formatter));
        }

        _formatters.AddOrUpdate(formatter.SupportedOperation, formatter, (_, _) => formatter);
    }

    /// <summary>
    /// Formats an error message using the appropriate formatter for the operation type.
    /// </summary>
    /// <param name="type">The type of operation that failed.</param>
    /// <param name="result">The validation result containing error information.</param>
    /// <param name="inputShapes">The shapes of the input tensors.</param>
    /// <returns>A formatted error message, or a generic message if no formatter is registered.</returns>
    public string FormatError(OperationType type, ValidationResult result, params long[][] inputShapes)
    {
        if (_formatters.TryGetValue(type, out var formatter))
        {
            return formatter.FormatError(result, inputShapes);
        }

        return FormatGenericError(type, result, inputShapes);
    }

    /// <summary>
    /// Gets suggested fixes using the appropriate formatter for the operation type.
    /// </summary>
    /// <param name="type">The type of operation that failed.</param>
    /// <param name="result">The validation result containing error information.</param>
    /// <returns>A list of suggested fixes, or an empty list if no formatter is registered.</returns>
    public List<string> GetSuggestions(OperationType type, ValidationResult result)
    {
        if (_formatters.TryGetValue(type, out var formatter))
        {
            return formatter.GenerateSuggestions(result);
        }

        return new List<string>();
    }

    /// <summary>
    /// Formats a generic error message when no specific formatter is available.
    /// </summary>
    private string FormatGenericError(OperationType type, ValidationResult result, long[][] inputShapes)
    {
        var shapes = inputShapes.Length > 0
            ? string.Join(", ", inputShapes.Select(shape => $"[{string.Join(", ", shape)}]"))
            : "N/A";

        return $"{type}: Validation failed for input shapes {shapes}. {result.ErrorMessage}";
    }
}
