using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Diagnostics
{
    /// <summary>
    /// Defines the shape requirements for an operation, including input count, expected dimensions,
    /// and dimension constraints used for validation.
    /// </summary>
    public class OperationShapeRequirements
    {
        /// <summary>
        /// Gets or sets the number of input tensors required by the operation.
        /// </summary>
        public int InputCount { get; set; }

        /// <summary>
        /// Gets or sets the expected dimension count for each input tensor.
        /// For example, new[] { 2, 2 } for matrix multiply (2D x 2D).
        /// </summary>
        public int[] ExpectedDimensions { get; set; }

        /// <summary>
        /// Gets or sets dimension constraints for the operation.
        /// Key: input tensor index, Value: dictionary of dimension index to constraint.
        /// Example: { [0, 1]: (1, "must_match", [1, 0]) }
        /// Means: input[0].dim[1] must match input[1].dim[0]
        /// </summary>
        public Dictionary<int, Dictionary<int, DimensionConstraint>> DimensionConstraints { get; set; }

        /// <summary>
        /// Gets or sets a human-readable description of the shape requirements.
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Gets or sets a format string for generating error messages.
        /// </summary>
        public string ErrorMessageFormat { get; set; }

        /// <summary>
        /// Gets or sets a custom validation function for complex validation logic.
        /// </summary>
        public Func<IEnumerable<long[]>, IDictionary<string, object>, ValidationResult> CustomValidator { get; set; }

        /// <summary>
        /// Initializes a new instance of OperationShapeRequirements with default values.
        /// </summary>
        public OperationShapeRequirements()
        {
            DimensionConstraints = new Dictionary<int, Dictionary<int, DimensionConstraint>>();
            CustomValidator = null;
        }

        /// <summary>
        /// Adds a dimension constraint for a specific input tensor and dimension.
        /// </summary>
        /// <param name="inputIndex">The index of the input tensor.</param>
        /// <param name="dimensionIndex">The index of the dimension within the tensor.</param>
        /// <param name="constraint">The constraint to apply.</param>
        public void AddConstraint(int inputIndex, int dimensionIndex, DimensionConstraint constraint)
        {
            if (!DimensionConstraints.ContainsKey(inputIndex))
            {
                DimensionConstraints[inputIndex] = new Dictionary<int, DimensionConstraint>();
            }
            DimensionConstraints[inputIndex][dimensionIndex] = constraint;
        }

        /// <summary>
        /// Validates input shapes against the operation's requirements.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <param name="operationParameters">Optional operation parameters for validation.</param>
        /// <returns>A ValidationResult indicating whether validation passed or failed.</returns>
        public ValidationResult ValidateShapes(IEnumerable<long[]> inputShapes, IDictionary<string, object> operationParameters = null)
        {
            var result = new ValidationResult();

            // Convert to array for indexed access
            var shapes = inputShapes.ToArray();

            // Validate input count
            if (shapes.Length != InputCount)
            {
                result.AddError($"Expected {InputCount} inputs, but got {shapes.Length}");
                return result;
            }

            // Validate dimension counts
            if (ExpectedDimensions != null)
            {
                for (int i = 0; i < shapes.Length; i++)
                {
                    if (i < ExpectedDimensions.Length && ExpectedDimensions[i] != shapes[i].Length)
                    {
                        result.AddError($"Input {i}: Expected {ExpectedDimensions[i]} dimensions, but got {shapes[i].Length}");
                    }
                }
            }

            // Validate dimension constraints
            if (DimensionConstraints != null && DimensionConstraints.Count > 0)
            {
                foreach (var inputConstraint in DimensionConstraints)
                {
                    int inputIndex = inputConstraint.Key;
                    var dimConstraints = inputConstraint.Value;

                    if (inputIndex >= shapes.Length)
                    {
                        result.AddError($"Cannot validate constraints for input {inputIndex}: input does not exist");
                        continue;
                    }

                    var shape = shapes[inputIndex];

                    foreach (var dimConstraint in dimConstraints)
                    {
                        int dimIndex = dimConstraint.Key;
                        var constraint = dimConstraint.Value;

                        if (dimIndex >= shape.Length)
                        {
                            result.AddError($"Input {inputIndex}: Dimension {dimIndex} does not exist");
                            continue;
                        }

                        long value = shape[dimIndex];
                        long? targetValue = null;

                        if (constraint.Type == DimensionConstraint.ConstraintType.MustMatch ||
                            constraint.Type == DimensionConstraint.ConstraintType.MustDivide)
                        {
                            if (constraint.TargetInputIndex.HasValue && constraint.TargetDimensionIndex.HasValue)
                            {
                                int targetInputIdx = constraint.TargetInputIndex.Value;
                                int targetDimIdx = constraint.TargetDimensionIndex.Value;

                                if (targetInputIdx >= shapes.Length)
                                {
                                    result.AddError($"Constraint on input {inputIndex} dim {dimIndex} references non-existent input {targetInputIdx}");
                                    continue;
                                }

                                if (targetDimIdx >= shapes[targetInputIdx].Length)
                                {
                                    result.AddError($"Constraint on input {inputIndex} dim {dimIndex} references non-existent dimension {targetDimIdx} in input {targetInputIdx}");
                                    continue;
                                }

                                targetValue = shapes[targetInputIdx][targetDimIdx];
                            }
                        }

                        if (!constraint.Validate(value, targetValue))
                        {
                            string errorMessage;
                            if (!string.IsNullOrEmpty(ErrorMessageFormat))
                            {
                                errorMessage = string.Format(ErrorMessageFormat, value, targetValue ?? "");
                            }
                            else
                            {
                                errorMessage = $"Input {inputIndex}, Dimension {dimIndex}: value {value} does not satisfy constraint {constraint}";
                            }

                            result.AddError(errorMessage);
                        }
                    }
                }
            }

            // Run custom validator if provided
            if (CustomValidator != null && result.IsValid)
            {
                var customResult = CustomValidator(shapes, operationParameters);
                if (!customResult.IsValid)
                {
                    result.IsValid = false;
                    if (customResult.Errors != null)
                    {
                        foreach (var error in customResult.Errors)
                        {
                            result.AddError(error);
                        }
                    }
                }
                if (customResult.Warnings != null)
                {
                    foreach (var warning in customResult.Warnings)
                    {
                        result.AddWarning(warning);
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Returns a string representation of the operation shape requirements.
        /// </summary>
        /// <returns>A description of the requirements.</returns>
        public override string ToString()
        {
            var parts = new List<string>();
            parts.Add($"Inputs: {InputCount}");

            if (ExpectedDimensions != null && ExpectedDimensions.Length > 0)
            {
                parts.Add($"ExpectedDimensions: [{string.Join(", ", ExpectedDimensions)}]");
            }

            if (!string.IsNullOrEmpty(Description))
            {
                parts.Add($"Description: {Description}");
            }

            return string.Join(", ", parts);
        }
    }
}
