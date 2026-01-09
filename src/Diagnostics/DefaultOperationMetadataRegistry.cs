using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.Core;

namespace MLFramework.Diagnostics
{
    /// <summary>
    /// Default implementation of IOperationMetadataRegistry that pre-registers common ML operations.
    /// Thread-safe using ConcurrentDictionary for operations.
    /// </summary>
    public class DefaultOperationMetadataRegistry : IOperationMetadataRegistry
    {
        private readonly Dictionary<OperationType, OperationShapeRequirements> _registry;
        private readonly object _lock = new object();

        /// <summary>
        /// Gets the singleton instance of the registry.
        /// </summary>
        public static DefaultOperationMetadataRegistry Instance { get; } = new DefaultOperationMetadataRegistry();

        /// <summary>
        /// Initializes a new instance of DefaultOperationMetadataRegistry with pre-registered operations.
        /// </summary>
        public DefaultOperationMetadataRegistry()
        {
            _registry = new Dictionary<OperationType, OperationShapeRequirements>();
            RegisterPredefinedOperations();
        }

        /// <summary>
        /// Registers shape requirements for an operation type.
        /// </summary>
        /// <param name="operationType">The type of operation to register.</param>
        /// <param name="requirements">The shape requirements for the operation.</param>
        public void RegisterOperation(OperationType operationType, OperationShapeRequirements requirements)
        {
            if (requirements == null)
            {
                throw new ArgumentNullException(nameof(requirements));
            }

            lock (_lock)
            {
                _registry[operationType] = requirements;
            }
        }

        /// <summary>
        /// Gets the shape requirements for a registered operation.
        /// </summary>
        /// <param name="operationType">The type of operation to get requirements for.</param>
        /// <returns>The shape requirements for the operation, or null if not registered.</returns>
        public OperationShapeRequirements GetRequirements(OperationType operationType)
        {
            lock (_lock)
            {
                return _registry.TryGetValue(operationType, out var requirements) ? requirements : null;
            }
        }

        /// <summary>
        /// Checks if an operation type is registered in the registry.
        /// </summary>
        /// <param name="operationType">The type of operation to check.</param>
        /// <returns>True if the operation is registered, false otherwise.</returns>
        public bool IsRegistered(OperationType operationType)
        {
            lock (_lock)
            {
                return _registry.ContainsKey(operationType);
            }
        }

        /// <summary>
        /// Validates input tensor shapes against an operation's requirements.
        /// </summary>
        /// <param name="operationType">The type of operation to validate against.</param>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <param name="operationParameters">Optional operation parameters used in validation.</param>
        /// <returns>A ValidationResult indicating whether the shapes are valid.</returns>
        public ValidationResult ValidateShapes(OperationType operationType, IEnumerable<long[]> inputShapes, System.Collections.IDictionary<string, object> operationParameters = null)
        {
            var requirements = GetRequirements(operationType);

            if (requirements == null)
            {
                return ValidationResult.Failure($"Operation type '{operationType}' is not registered in the metadata registry.");
            }

            return requirements.ValidateShapes(inputShapes, operationParameters);
        }

        /// <summary>
        /// Registers all predefined ML operations with their shape requirements.
        /// </summary>
        private void RegisterPredefinedOperations()
        {
            RegisterMatrixMultiply();
            RegisterConv2D();
            RegisterMaxPool2D();
            RegisterAveragePool2D();
            RegisterConcat();
            RegisterStack();
            RegisterFlatten();
            RegisterReshape();
            RegisterTranspose();
            RegisterBroadcast();
        }

        /// <summary>
        /// Registers MatrixMultiply operation requirements.
        /// </summary>
        private void RegisterMatrixMultiply()
        {
            var requirements = new OperationShapeRequirements
            {
                InputCount = 2,
                ExpectedDimensions = new[] { 2, 2 },
                DimensionConstraints = new Dictionary<int, Dictionary<int, DimensionConstraint>>
                {
                    {
                        0, new Dictionary<int, DimensionConstraint>
                        {
                            { 1, new DimensionConstraint
                                {
                                    Type = DimensionConstraint.ConstraintType.MustMatch,
                                    TargetInputIndex = 1,
                                    TargetDimensionIndex = 0
                                }
                            }
                        }
                    }
                },
                Description = "Matrix multiplication: [batch, m] × [m, n] → [batch, n]",
                ErrorMessageFormat = "Dimension 1 of input ({0}) does not match dimension 0 of weight ({1})"
            };

            RegisterOperation(OperationType.MatrixMultiply, requirements);
        }

        /// <summary>
        /// Registers Conv2D operation requirements with custom validator.
        /// </summary>
        private void RegisterConv2D()
        {
            var requirements = new OperationShapeRequirements
            {
                InputCount = 2,
                ExpectedDimensions = new[] { 4, 4 }, // NCHW format
                Description = "2D Convolution: [N, C_in, H, W] × [C_out, C_in, kH, kW] → [N, C_out, H_out, W_out]",
                CustomValidator = ValidateConv2DShapes
            };

            RegisterOperation(OperationType.Conv2D, requirements);
        }

        /// <summary>
        /// Custom validator for Conv2D operations.
        /// </summary>
        private ValidationResult ValidateConv2DShapes(IEnumerable<long[]> inputShapes, IDictionary<string, object> operationParameters)
        {
            var result = new ValidationResult();
            var shapes = inputShapes.ToArray();

            var inputShape = shapes[0]; // [N, C_in, H, W]
            var weightShape = shapes[1]; // [C_out, C_in, kH, kW]

            // Validate input channel match
            if (inputShape[1] != weightShape[1])
            {
                result.AddError($"Input channels ({inputShape[1]}) must match weight in-channels ({weightShape[1]})");
            }

            // Validate kernel size
            if (weightShape[2] < 1 || weightShape[3] < 1)
            {
                result.AddError("Kernel dimensions must be positive");
            }

            return result;
        }

        /// <summary>
        /// Registers MaxPool2D operation requirements.
        /// </summary>
        private void RegisterMaxPool2D()
        {
            var requirements = new OperationShapeRequirements
            {
                InputCount = 1,
                ExpectedDimensions = new[] { 4 }, // NCHW format
                Description = "2D Max Pooling: [N, C, H, W] → [N, C, H_out, W_out]"
            };

            RegisterOperation(OperationType.MaxPool2D, requirements);
        }

        /// <summary>
        /// Registers AveragePool2D operation requirements.
        /// </summary>
        private void RegisterAveragePool2D()
        {
            var requirements = new OperationShapeRequirements
            {
                InputCount = 1,
                ExpectedDimensions = new[] { 4 }, // NCHW format
                Description = "2D Average Pooling: [N, C, H, W] → [N, C, H_out, W_out]"
            };

            RegisterOperation(OperationType.AveragePool2D, requirements);
        }

        /// <summary>
        /// Registers Concat operation requirements.
        /// </summary>
        private void RegisterConcat()
        {
            var requirements = new OperationShapeRequirements
            {
                InputCount = -1, // Variable number of inputs
                Description = "Concatenate tensors along a specified dimension",
                CustomValidator = ValidateConcatShapes
            };

            RegisterOperation(OperationType.Concat, requirements);
        }

        /// <summary>
        /// Custom validator for Concat operations.
        /// </summary>
        private ValidationResult ValidateConcatShapes(IEnumerable<long[]> inputShapes, IDictionary<string, object> operationParameters)
        {
            var result = new ValidationResult();
            var shapes = inputShapes.ToArray();

            if (shapes.Length < 2)
            {
                result.AddError("Concat requires at least 2 input tensors");
                return result;
            }

            int axis = 0;
            if (operationParameters != null && operationParameters.ContainsKey("axis"))
            {
                axis = Convert.ToInt32(operationParameters["axis"]);
            }

            var firstShape = shapes[0];

            // Check that all inputs have same rank
            foreach (var shape in shapes)
            {
                if (shape.Length != firstShape.Length)
                {
                    result.AddError("All input tensors must have the same number of dimensions for concatenation");
                    return result;
                }
            }

            // Check that all dimensions except the concat axis match
            for (int i = 0; i < firstShape.Length; i++)
            {
                if (i == axis) continue;

                for (int j = 1; j < shapes.Length; j++)
                {
                    if (shapes[j][i] != firstShape[i])
                    {
                        result.AddError($"Dimension {i} mismatch between input 0 ({firstShape[i]}) and input {j} ({shapes[j][i]})");
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Registers Stack operation requirements.
        /// </summary>
        private void RegisterStack()
        {
            var requirements = new OperationShapeRequirements
            {
                InputCount = -1, // Variable number of inputs
                Description = "Stack tensors along a new dimension",
                CustomValidator = ValidateStackShapes
            };

            RegisterOperation(OperationType.Stack, requirements);
        }

        /// <summary>
        /// Custom validator for Stack operations.
        /// </summary>
        private ValidationResult ValidateStackShapes(IEnumerable<long[]> inputShapes, IDictionary<string, object> operationParameters)
        {
            var result = new ValidationResult();
            var shapes = inputShapes.ToArray();

            if (shapes.Length < 2)
            {
                result.AddError("Stack requires at least 2 input tensors");
                return result;
            }

            var firstShape = shapes[0];

            // All input tensors must have identical shapes
            for (int i = 1; i < shapes.Length; i++)
            {
                if (shapes[i].Length != firstShape.Length)
                {
                    result.AddError($"Input {i} has different rank ({shapes[i].Length}) than input 0 ({firstShape.Length})");
                    continue;
                }

                for (int j = 0; j < firstShape.Length; j++)
                {
                    if (shapes[i][j] != firstShape[j])
                    {
                        result.AddError($"Dimension {j} mismatch between input 0 ({firstShape[j]}) and input {i} ({shapes[i][j]})");
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Registers Flatten operation requirements.
        /// </summary>
        private void RegisterFlatten()
        {
            var requirements = new OperationShapeRequirements
            {
                InputCount = 1,
                Description = "Flatten a tensor from start_dim to end_dim"
            };

            RegisterOperation(OperationType.Flatten, requirements);
        }

        /// <summary>
        /// Registers Reshape operation requirements.
        /// </summary>
        private void RegisterReshape()
        {
            var requirements = new OperationShapeRequirements
            {
                InputCount = 1,
                Description = "Reshape a tensor to a new shape",
                CustomValidator = ValidateReshapeShapes
            };

            RegisterOperation(OperationType.Reshape, requirements);
        }

        /// <summary>
        /// Custom validator for Reshape operations.
        /// </summary>
        private ValidationResult ValidateReshapeShapes(IEnumerable<long[]> inputShapes, IDictionary<string, object> operationParameters)
        {
            var result = new ValidationResult();
            var shapes = inputShapes.ToArray();

            if (operationParameters == null || !operationParameters.ContainsKey("shape"))
            {
                result.AddError("Reshape operation requires 'shape' parameter");
                return result;
            }

            var targetShape = operationParameters["shape"] as long[];
            if (targetShape == null)
            {
                result.AddError("'shape' parameter must be a long array");
                return result;
            }

            // Check that at most one dimension is -1
            int negativeDimCount = targetShape.Count(dim => dim == -1);
            if (negativeDimCount > 1)
            {
                result.AddError("Reshape can have at most one dimension set to -1");
            }

            // Calculate total elements
            long totalElements = 1;
            foreach (var dim in shapes[0])
            {
                totalElements *= dim;
            }

            long knownSize = 1;
            foreach (var dim in targetShape)
            {
                if (dim > 0)
                {
                    knownSize *= dim;
                }
            }

            if (negativeDimCount == 0 && knownSize != totalElements)
            {
                result.AddError($"Reshape size mismatch: input has {totalElements} elements, target shape has {knownSize} elements");
            }
            else if (negativeDimCount == 1 && totalElements % knownSize != 0)
            {
                result.AddError($"Cannot infer dimension: input has {totalElements} elements, which is not divisible by {knownSize}");
            }

            return result;
        }

        /// <summary>
        /// Registers Transpose operation requirements.
        /// </summary>
        private void RegisterTranspose()
        {
            var requirements = new OperationShapeRequirements
            {
                InputCount = 1,
                Description = "Transpose tensor dimensions"
            };

            RegisterOperation(OperationType.Transpose, requirements);
        }

        /// <summary>
        /// Registers Broadcast operation requirements.
        /// </summary>
        private void RegisterBroadcast()
        {
            var requirements = new OperationShapeRequirements
            {
                InputCount = -1, // Variable number of inputs
                Description = "Broadcast tensors to a common shape",
                CustomValidator = ValidateBroadcastShapes
            };

            RegisterOperation(OperationType.Broadcast, requirements);
        }

        /// <summary>
        /// Custom validator for Broadcast operations.
        /// </summary>
        private ValidationResult ValidateBroadcastShapes(IEnumerable<long[]> inputShapes, IDictionary<string, object> operationParameters)
        {
            var result = new ValidationResult();
            var shapes = inputShapes.ToArray();

            if (shapes.Length < 2)
            {
                result.AddError("Broadcast requires at least 2 input tensors");
                return result;
            }

            // Find maximum rank
            int maxRank = shapes.Max(s => s.Length);

            // Validate broadcast compatibility
            for (int i = 0; i < shapes.Length; i++)
            {
                for (int j = i + 1; j < shapes.Length; j++)
                {
                    if (!AreShapesBroadcastCompatible(shapes[i], shapes[j]))
                    {
                        result.AddError($"Shapes {ShapeToString(shapes[i])} and {ShapeToString(shapes[j])} are not broadcast compatible");
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Checks if two shapes are broadcast compatible.
        /// </summary>
        private bool AreShapesBroadcastCompatible(long[] shape1, long[] shape2)
        {
            int maxRank = Math.Max(shape1.Length, shape2.Length);

            for (int i = 0; i < maxRank; i++)
            {
                int idx1 = shape1.Length - 1 - i;
                int idx2 = shape2.Length - 1 - i;

                long dim1 = idx1 >= 0 ? shape1[idx1] : 1;
                long dim2 = idx2 >= 0 ? shape2[idx2] : 1;

                if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Converts a shape array to a string representation.
        /// </summary>
        private string ShapeToString(long[] shape)
        {
            return "[" + string.Join(", ", shape) + "]";
        }
    }
}
