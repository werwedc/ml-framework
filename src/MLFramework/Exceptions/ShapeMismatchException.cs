using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MLFramework.Core;

namespace MLFramework.Exceptions
{
    /// <summary>
    /// Custom exception that provides rich diagnostic information for tensor shape mismatches.
    /// Includes detailed information about the operation, input shapes, expected shapes,
    /// and suggested fixes to help developers quickly diagnose and resolve shape-related errors.
    /// </summary>
    public class ShapeMismatchException : Exception
    {
        /// <summary>
        /// Gets the name of the layer/module where the error occurred.
        /// </summary>
        public string LayerName { get; }

        /// <summary>
        /// Gets the type of operation that caused the shape mismatch.
        /// </summary>
        public OperationType OperationType { get; }

        /// <summary>
        /// Gets the shapes of the input tensors that caused the error.
        /// </summary>
        public IReadOnlyList<long[]> InputShapes { get; }

        /// <summary>
        /// Gets the expected shapes for the operation.
        /// </summary>
        public IReadOnlyList<long[]> ExpectedShapes { get; }

        /// <summary>
        /// Gets a human-readable description of the problem.
        /// </summary>
        public string ProblemDescription { get; }

        /// <summary>
        /// Gets a list of suggested fixes for the shape mismatch.
        /// </summary>
        public IReadOnlyList<string> SuggestedFixes { get; }

        /// <summary>
        /// Gets the batch size if applicable.
        /// </summary>
        public long? BatchSize { get; }

        /// <summary>
        /// Gets context about the previous layer, if available.
        /// </summary>
        public string PreviousLayerContext { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ShapeMismatchException"/> class with detailed diagnostic information.
        /// </summary>
        /// <param name="layerName">Name of the layer/module where the error occurred.</param>
        /// <param name="operationType">Type of operation that caused the shape mismatch.</param>
        /// <param name="inputShapes">Shapes of the input tensors.</param>
        /// <param name="expectedShapes">Expected shapes for the operation.</param>
        /// <param name="problemDescription">Human-readable description of the problem.</param>
        /// <param name="suggestedFixes">Optional list of suggested fixes.</param>
        /// <param name="batchSize">Optional batch size if applicable.</param>
        /// <param name="previousLayerContext">Optional context about the previous layer.</param>
        public ShapeMismatchException(
            string layerName,
            OperationType operationType,
            IEnumerable<long[]> inputShapes,
            IEnumerable<long[]> expectedShapes,
            string problemDescription,
            IEnumerable<string>? suggestedFixes = null,
            long? batchSize = null,
            string? previousLayerContext = null)
            : base(GenerateMessage(layerName, operationType, inputShapes, expectedShapes))
        {
            LayerName = layerName ?? throw new ArgumentNullException(nameof(layerName));
            OperationType = operationType;
            InputShapes = inputShapes?.ToList()?.AsReadOnly() ?? throw new ArgumentNullException(nameof(inputShapes));
            ExpectedShapes = expectedShapes?.ToList()?.AsReadOnly();
            ProblemDescription = problemDescription ?? throw new ArgumentNullException(nameof(problemDescription));
            SuggestedFixes = suggestedFixes?.ToList()?.AsReadOnly();
            BatchSize = batchSize;
            PreviousLayerContext = previousLayerContext;
        }

        /// <summary>
        /// Gets a formatted diagnostic report with detailed information about the shape mismatch.
        /// </summary>
        /// <returns>A formatted string containing all diagnostic information.</returns>
        public string GetDiagnosticReport()
        {
            var sb = new StringBuilder();

            sb.AppendLine($"ShapeMismatchException: {GenerateOperationErrorMessage()}");

            sb.AppendLine();

            // Input shapes
            if (InputShapes != null && InputShapes.Count > 0)
            {
                string label1 = GetShapeLabel(OperationType, 0);
                sb.AppendLine($"{label1} shape:    [{string.Join(", ", InputShapes[0])}]");
            }

            if (InputShapes != null && InputShapes.Count > 1)
            {
                string label2 = GetShapeLabel(OperationType, 1);
                sb.AppendLine($"{label2} shape:   [{string.Join(", ", InputShapes[1])}]");
            }

            sb.AppendLine();

            // Expected format
            string expectedFormat = GetExpectedFormatString();
            sb.AppendLine("Expected:");
            sb.AppendLine($"                {expectedFormat}");
            sb.AppendLine();

            // Problem description
            sb.AppendLine($"Problem: {ProblemDescription}");
            sb.AppendLine();

            // Context
            if (!string.IsNullOrEmpty(LayerName) || BatchSize.HasValue || !string.IsNullOrEmpty(PreviousLayerContext))
            {
                sb.AppendLine("Context:");
                if (!string.IsNullOrEmpty(LayerName))
                {
                    sb.AppendLine($"- Layer: {LayerName} ({OperationType})");
                }
                if (BatchSize.HasValue)
                {
                    sb.AppendLine($"- Batch size: {BatchSize.Value}");
                }
                if (!string.IsNullOrEmpty(PreviousLayerContext))
                {
                    sb.AppendLine($"- Previous layer output: {PreviousLayerContext}");
                }
                sb.AppendLine();
            }

            // Suggested fixes
            if (SuggestedFixes != null && SuggestedFixes.Count > 0)
            {
                sb.AppendLine("Suggested fixes:");
                for (int i = 0; i < SuggestedFixes.Count; i++)
                {
                    sb.AppendLine($"{i + 1}. {SuggestedFixes[i]}");
                }
            }

            return sb.ToString().Trim();
        }

        /// <summary>
        /// Visualizes the shape flow (not yet implemented).
        /// </summary>
        /// <param name="outputPath">Path where the visualization should be saved.</param>
        public void VisualizeShapeFlow(string outputPath)
        {
            throw new NotImplementedException("Shape visualization not yet implemented");
        }

        private static string GenerateMessage(
            string layerName,
            OperationType operationType,
            IEnumerable<long[]> inputShapes,
            IEnumerable<long[]> expectedShapes)
        {
            return $"Shape mismatch in layer '{layerName}' during {operationType} operation";
        }

        private string GenerateOperationErrorMessage()
        {
            switch (OperationType)
            {
                case OperationType.MatrixMultiply:
                    return $"Matrix multiplication failed in layer '{LayerName}'";
                case OperationType.Conv2D:
                    return $"Conv2D operation failed in layer '{LayerName}'";
                case OperationType.Conv1D:
                    return $"Conv1D operation failed in layer '{LayerName}'";
                case OperationType.Concat:
                    return $"Concatenation failed in layer '{LayerName}'";
                case OperationType.Stack:
                    return $"Stack operation failed in layer '{LayerName}'";
                case OperationType.Reshape:
                    return $"Reshape operation failed in layer '{LayerName}'";
                case OperationType.Transpose:
                    return $"Transpose operation failed in layer '{LayerName}'";
                case OperationType.Flatten:
                    return $"Flatten operation failed in layer '{LayerName}'";
                case OperationType.MaxPool2D:
                    return $"MaxPool2D operation failed in layer '{LayerName}'";
                case OperationType.AveragePool2D:
                    return $"AveragePool2D operation failed in layer '{LayerName}'";
                default:
                    return $"Operation failed in layer '{LayerName}'";
            }
        }

        private static string GetShapeLabel(OperationType operation, int index)
        {
            switch (operation)
            {
                case OperationType.MatrixMultiply:
                    return index == 0 ? "Input" : "Weight";
                case OperationType.Conv2D:
                case OperationType.Conv1D:
                    return index == 0 ? "Input" : "Kernel";
                case OperationType.Concat:
                case OperationType.Stack:
                    return $"Input {index}";
                default:
                    return $"Tensor {index}";
            }
        }

        private string GetExpectedFormatString()
        {
            switch (OperationType)
            {
                case OperationType.MatrixMultiply:
                    return "[batch_size, input_features] × [input_features, output_features]";
                case OperationType.Conv2D:
                    return "[N, C_in, H, W] × [C_out, C_in, kH, kW]";
                case OperationType.Conv1D:
                    return "[N, C_in, L] × [C_out, C_in, k]";
                case OperationType.MaxPool2D:
                case OperationType.AveragePool2D:
                    return "[N, C, H, W] → [N, C, H_out, W_out]";
                case OperationType.Concat:
                case OperationType.Stack:
                    return "All tensors must have matching dimensions except at concatenation/stack axis";
                case OperationType.Reshape:
                    return "Total number of elements must remain the same";
                case OperationType.Transpose:
                    return "Dimensions must be valid for transpose operation";
                case OperationType.Flatten:
                    return "Flattens specified dimensions";
                default:
                    return "Shapes must be compatible for this operation";
            }
        }
    }
}
