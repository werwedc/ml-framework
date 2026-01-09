using System.Text;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Contains detailed shape diagnostics information for an operation.
/// </summary>
public class ShapeDiagnosticsInfo
{
    /// <summary>
    /// The type of operation being diagnosed.
    /// </summary>
    public OperationType OperationType { get; set; }

    /// <summary>
    /// The name of the layer (optional).
    /// </summary>
    public string LayerName { get; set; }

    /// <summary>
    /// The shapes of input tensors.
    /// </summary>
    public long[][] InputShapes { get; set; }

    /// <summary>
    /// The expected shapes for the operation.
    /// </summary>
    public long[][] ExpectedShapes { get; set; }

    /// <summary>
    /// The actual inferred output shape.
    /// </summary>
    public long[] ActualOutputShape { get; set; }

    /// <summary>
    /// Whether the shapes are valid.
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// List of error messages.
    /// </summary>
    public List<string> Errors { get; set; }

    /// <summary>
    /// List of warning messages.
    /// </summary>
    public List<string> Warnings { get; set; }

    /// <summary>
    /// Human-readable description of requirements.
    /// </summary>
    public string RequirementsDescription { get; set; }

    /// <summary>
    /// Name of the previous layer (for contextual diagnostics).
    /// </summary>
    public string PreviousLayerName { get; set; }

    /// <summary>
    /// Shape of the previous layer's output (for contextual diagnostics).
    /// </summary>
    public long[] PreviousLayerShape { get; set; }

    /// <summary>
    /// Gets a formatted report of the shape diagnostics.
    /// </summary>
    public string GetFormattedReport()
    {
        var sb = new StringBuilder();

        sb.AppendLine($"Shape Diagnostics for layer '{LayerName}'");
        sb.AppendLine($"Operation: {OperationType}");
        sb.AppendLine($"Valid: {IsValid}");
        sb.AppendLine();

        if (InputShapes != null && InputShapes.Length > 0)
        {
            sb.AppendLine("Input Shapes:");
            for (int i = 0; i < InputShapes.Length; i++)
            {
                sb.AppendLine($"  Input {i}: [{string.Join(", ", InputShapes[i])}]");
            }
            sb.AppendLine();
        }

        if (ExpectedShapes != null && ExpectedShapes.Length > 0)
        {
            sb.AppendLine("Expected Shapes:");
            for (int i = 0; i < ExpectedShapes.Length; i++)
            {
                sb.AppendLine($"  Expected {i}: [{string.Join(", ", ExpectedShapes[i])}]");
            }
            sb.AppendLine();
        }

        if (ActualOutputShape != null)
        {
            sb.AppendLine($"Output Shape: [{string.Join(", ", ActualOutputShape)}]");
            sb.AppendLine();
        }

        if (RequirementsDescription != null)
        {
            sb.AppendLine($"Requirements: {RequirementsDescription}");
            sb.AppendLine();
        }

        if (!IsValid && Errors != null && Errors.Count > 0)
        {
            sb.AppendLine("Errors:");
            foreach (var error in Errors)
            {
                sb.AppendLine($"  - {error}");
            }
            sb.AppendLine();
        }

        if (Warnings != null && Warnings.Count > 0)
        {
            sb.AppendLine("Warnings:");
            foreach (var warning in Warnings)
            {
                sb.AppendLine($"  - {warning}");
            }
            sb.AppendLine();
        }

        if (PreviousLayerName != null)
        {
            sb.AppendLine("Context:");
            sb.AppendLine($"  Previous layer: {PreviousLayerName}");
            if (PreviousLayerShape != null)
            {
                sb.AppendLine($"  Previous output: [{string.Join(", ", PreviousLayerShape)}]");
            }
        }

        return sb.ToString();
    }
}
