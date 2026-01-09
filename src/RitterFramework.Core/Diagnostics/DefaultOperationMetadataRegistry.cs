using System.Collections.Concurrent;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Default implementation of IOperationMetadataRegistry with pre-registered operations.
/// </summary>
public class DefaultOperationMetadataRegistry : IOperationMetadataRegistry
{
    private readonly ConcurrentDictionary<OperationType, OperationShapeRequirements> _registry;

    public DefaultOperationMetadataRegistry()
    {
        _registry = new ConcurrentDictionary<OperationType, OperationShapeRequirements>();
        RegisterDefaultOperations();
    }

    /// <inheritdoc/>
    public void RegisterOperation(
        OperationType operationType,
        OperationShapeRequirements requirements)
    {
        _registry[operationType] = requirements;
    }

    /// <inheritdoc/>
    public OperationShapeRequirements GetRequirements(OperationType operationType)
    {
        if (!_registry.TryGetValue(operationType, out var requirements))
        {
            return new OperationShapeRequirements
            {
                Description = "No requirements registered for this operation"
            };
        }

        return requirements;
    }

    /// <inheritdoc/>
    public bool IsRegistered(OperationType operationType)
    {
        return _registry.ContainsKey(operationType);
    }

    /// <inheritdoc/>
    public ValidationResult ValidateShapes(
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters = null)
    {
        var shapes = inputShapes.ToArray();

        if (!_registry.TryGetValue(operationType, out var requirements))
        {
            return ValidationResult.Failure($"Operation {operationType} is not registered");
        }

        // Check input count
        if (shapes.Length != requirements.InputCount)
        {
            return ValidationResult.Failure(
                $"Expected {requirements.InputCount} inputs, got {shapes.Length}");
        }

        // Check dimension counts
        if (requirements.ExpectedDimensions != null && shapes.Length == requirements.ExpectedDimensions.Length)
        {
            for (int i = 0; i < shapes.Length; i++)
            {
                if (shapes[i].Length != requirements.ExpectedDimensions[i])
                {
                    return ValidationResult.Failure(
                        $"Input {i} has {shapes[i].Length} dimensions, expected {requirements.ExpectedDimensions[i]}");
                }
            }
        }

        // Check dimension constraints
        if (requirements.DimensionConstraints != null)
        {
            foreach (var inputConstraints in requirements.DimensionConstraints)
            {
                int inputIndex = inputConstraints.Key;
                var constraints = inputConstraints.Value;

                foreach (var constraintPair in constraints)
                {
                    int dimIndex = constraintPair.Key;
                    var constraint = constraintPair.Value;

                    if (!ValidateDimensionConstraint(
                        shapes,
                        inputIndex,
                        dimIndex,
                        constraint,
                        out string error))
                    {
                        return ValidationResult.Failure(error);
                    }
                }
            }
        }

        // Use custom validator if provided
        if (requirements.CustomValidator != null)
        {
            return requirements.CustomValidator(shapes, operationParameters);
        }

        return ValidationResult.Success();
    }

    private bool ValidateDimensionConstraint(
        long[][] inputShapes,
        int inputIndex,
        int dimIndex,
        DimensionConstraint constraint,
        out string error)
    {
        error = null;

        if (inputIndex >= inputShapes.Length || dimIndex >= inputShapes[inputIndex].Length)
        {
            error = $"Invalid constraint: input {inputIndex}, dimension {dimIndex}";
            return false;
        }

        long value = inputShapes[inputIndex][dimIndex];

        switch (constraint.Type)
        {
            case DimensionConstraint.ConstraintType.MustMatch:
                if (constraint.TargetInputIndex.HasValue && constraint.TargetDimensionIndex.HasValue)
                {
                    int targetInput = constraint.TargetInputIndex.Value;
                    int targetDim = constraint.TargetDimensionIndex.Value;

                    if (targetInput >= inputShapes.Length || targetDim >= inputShapes[targetInput].Length)
                    {
                        error = $"Invalid target: input {targetInput}, dimension {targetDim}";
                        return false;
                    }

                    long targetValue = inputShapes[targetInput][targetDim];
                    if (value != targetValue)
                    {
                        error = $"Dimension mismatch: input[{inputIndex}][{dimIndex}]={value}, input[{targetInput}][{targetDim}]={targetValue}";
                        return false;
                    }
                }
                break;

            case DimensionConstraint.ConstraintType.MustEqual:
                if (constraint.FixedValue.HasValue && value != constraint.FixedValue.Value)
                {
                    error = $"Dimension {dimIndex} must be {constraint.FixedValue.Value}, got {value}";
                    return false;
                }
                break;

            case DimensionConstraint.ConstraintType.MustBePositive:
                if (value <= 0)
                {
                    error = $"Dimension {dimIndex} must be positive, got {value}";
                    return false;
                }
                break;

            case DimensionConstraint.ConstraintType.MustBeMultipleOf:
                if (constraint.MultipleOf.HasValue && value % constraint.MultipleOf.Value != 0)
                {
                    error = $"Dimension {dimIndex} must be multiple of {constraint.MultipleOf.Value}, got {value}";
                    return false;
                }
                break;
        }

        return true;
    }

    private void RegisterDefaultOperations()
    {
        // Matrix Multiply
        RegisterOperation(OperationType.MatrixMultiply, new OperationShapeRequirements
        {
            InputCount = 2,
            ExpectedDimensions = new[] { 2, 2 },
            DimensionConstraints = new Dictionary<int, Dictionary<int, DimensionConstraint>>
            {
                {
                    0, new Dictionary<int, DimensionConstraint>
                    {
                        { 1, new DimensionConstraint { Type = DimensionConstraint.ConstraintType.MustMatch,
                                                        TargetInputIndex = 1,
                                                        TargetDimensionIndex = 0 } }
                    }
                }
            },
            Description = "Matrix multiplication: [batch, m] × [m, n] → [batch, n]",
            ErrorMessageFormat = "Dimension 1 of input ({0}) does not match dimension 0 of weight ({1})"
        });

        // Conv2D
        RegisterOperation(OperationType.Conv2D, new OperationShapeRequirements
        {
            InputCount = 2,
            ExpectedDimensions = new[] { 4, 4 }, // NCHW format
            Description = "2D Convolution: [N, C_in, H, W] × [C_out, C_in, kH, kW] → [N, C_out, H_out, W_out]"
        });

        // Linear
        RegisterOperation(OperationType.Linear, new OperationShapeRequirements
        {
            InputCount = 2,
            ExpectedDimensions = new[] { 2, 2 },
            Description = "Linear layer: [batch, in_features] × [out_features, in_features] → [batch, out_features]"
        });

        // Concat
        RegisterOperation(OperationType.Concat, new OperationShapeRequirements
        {
            InputCount = -1, // Variable
            Description = "Concatenation: Concatenate tensors along specified axis"
        });

        // Stack
        RegisterOperation(OperationType.Stack, new OperationShapeRequirements
        {
            InputCount = -1, // Variable
            Description = "Stack: Stack tensors along new dimension"
        });

        // Transpose
        RegisterOperation(OperationType.Transpose, new OperationShapeRequirements
        {
            InputCount = 1,
            Description = "Transpose: Transpose tensor dimensions"
        });

        // Reshape
        RegisterOperation(OperationType.Reshape, new OperationShapeRequirements
        {
            InputCount = 1,
            Description = "Reshape: Change tensor shape while keeping element count"
        });
    }
}
