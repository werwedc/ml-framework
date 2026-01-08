using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Represents a tensor that will be broadcast to a target shape.
    /// </summary>
    public class BroadcastedTensor
    {
        /// <summary>
        /// Gets the original tensor before broadcasting.
        /// </summary>
        public Tensor OriginalTensor { get; }

        /// <summary>
        /// Gets the broadcast shape (target shape).
        /// </summary>
        public SymbolicShape BroadcastShape { get; }

        /// <summary>
        /// Gets the broadcast plan that describes how broadcasting should be performed.
        /// </summary>
        public List<BroadcastingRule> BroadcastPlan { get; }

        /// <summary>
        /// Gets the original shape.
        /// </summary>
        public SymbolicShape OriginalShape { get; }

        /// <summary>
        /// Initializes a new instance of the BroadcastedTensor class.
        /// </summary>
        /// <param name="originalTensor">The original tensor to broadcast.</param>
        /// <param name="broadcastShape">The target shape to broadcast to.</param>
        /// <param name="broadcastPlan">The plan describing how to broadcast.</param>
        public BroadcastedTensor(
            Tensor originalTensor,
            SymbolicShape broadcastShape,
            List<BroadcastingRule> broadcastPlan)
        {
            OriginalTensor = originalTensor ?? throw new ArgumentNullException(nameof(originalTensor));
            BroadcastShape = broadcastShape ?? throw new ArgumentNullException(nameof(broadcastShape));
            BroadcastPlan = broadcastPlan ?? throw new ArgumentNullException(nameof(broadcastPlan));

            // Create symbolic shape from original tensor's concrete shape
            var originalDims = originalTensor.Shape
                .Select((d, i) => SymbolicDimensionFactory.CreateKnown($"dim_{i}", d))
                .ToArray();
            OriginalShape = new SymbolicShape(originalDims);
        }

        /// <summary>
        /// Initializes a new instance of the BroadcastedTensor class.
        /// </summary>
        /// <param name="originalTensor">The original tensor to broadcast.</param>
        /// <param name="originalShape">The original symbolic shape.</param>
        /// <param name="broadcastShape">The target shape to broadcast to.</param>
        /// <param name="broadcastPlan">The plan describing how to broadcast.</param>
        public BroadcastedTensor(
            Tensor originalTensor,
            SymbolicShape originalShape,
            SymbolicShape broadcastShape,
            List<BroadcastingRule> broadcastPlan)
        {
            OriginalTensor = originalTensor ?? throw new ArgumentNullException(nameof(originalTensor));
            OriginalShape = originalShape ?? throw new ArgumentNullException(nameof(originalShape));
            BroadcastShape = broadcastShape ?? throw new ArgumentNullException(nameof(broadcastShape));
            BroadcastPlan = broadcastPlan ?? throw new ArgumentNullException(nameof(broadcastPlan));
        }

        /// <summary>
        /// Materializes the broadcast by creating a new tensor with the broadcast shape.
        /// </summary>
        /// <returns>A new tensor with the broadcast shape.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the broadcast shape cannot be materialized (contains unknown dimensions).</exception>
        public Tensor Materialize()
        {
            // Convert symbolic shape to concrete
            if (!BroadcastShape.IsFullyKnown())
            {
                throw new InvalidOperationException(
                    $"Cannot materialize broadcast with unknown dimensions: {BroadcastShape}");
            }

            int[] broadcastShapeArray = BroadcastShape.ToConcrete();
            int[] originalShapeArray = OriginalTensor.Shape;

            // Check if broadcasting is actually needed
            if (originalShapeArray.SequenceEqual(broadcastShapeArray))
            {
                return OriginalTensor;
            }

            // Create the broadcasted tensor
            return CreateBroadcastedTensor(OriginalTensor, originalShapeArray, broadcastShapeArray);
        }

        /// <summary>
        /// Calculates the strides for the broadcasted tensor.
        /// </summary>
        /// <returns>An array of strides.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the broadcast shape cannot be materialized.</exception>
        public int[] GetStrides()
        {
            if (!BroadcastShape.IsFullyKnown())
            {
                throw new InvalidOperationException(
                    $"Cannot calculate strides with unknown dimensions: {BroadcastShape}");
            }

            int[] broadcastShapeArray = BroadcastShape.ToConcrete();
            int[] originalShapeArray = OriginalTensor.Shape;

            // Pad original shape to match broadcast shape rank
            int rankDiff = broadcastShapeArray.Length - originalShapeArray.Length;
            int[] paddedOriginalShape = new int[broadcastShapeArray.Length];
            for (int i = 0; i < rankDiff; i++)
            {
                paddedOriginalShape[i] = 1;
            }
            Array.Copy(originalShapeArray, 0, paddedOriginalShape, rankDiff, originalShapeArray.Length);

            // Calculate strides
            int[] strides = new int[broadcastShapeArray.Length];
            strides[broadcastShapeArray.Length - 1] = 1;

            for (int i = broadcastShapeArray.Length - 2; i >= 0; i--)
            {
                if (paddedOriginalShape[i] == 1 && broadcastShapeArray[i] > 1)
                {
                    // This dimension is being broadcast, stride is 0
                    strides[i] = 0;
                }
                else
                {
                    strides[i] = strides[i + 1] * broadcastShapeArray[i + 1];
                }
            }

            return strides;
        }

        /// <summary>
        /// Checks if this broadcasted tensor requires actual materialization.
        /// </summary>
        /// <returns>True if materialization is required; otherwise, false.</returns>
        public bool RequiresMaterialization()
        {
            return !OriginalShape.Equals(BroadcastShape);
        }

        /// <summary>
        /// Gets the broadcast source (identifies which dimensions are being broadcast).
        /// </summary>
        /// <returns>A list of dimension indices that are being broadcast.</returns>
        public List<int> GetBroadcastSources()
        {
            var sources = new List<int>();
            int[] originalShapeArray = OriginalTensor.Shape;
            int[] broadcastShapeArray = BroadcastShape.ToConcrete();

            // Pad original shape to match broadcast shape rank
            int rankDiff = broadcastShapeArray.Length - originalShapeArray.Length;
            int[] paddedOriginalShape = new int[broadcastShapeArray.Length];
            for (int i = 0; i < rankDiff; i++)
            {
                paddedOriginalShape[i] = 1;
            }
            Array.Copy(originalShapeArray, 0, paddedOriginalShape, rankDiff, originalShapeArray.Length);

            for (int i = 0; i < broadcastShapeArray.Length; i++)
            {
                if (paddedOriginalShape[i] == 1 && broadcastShapeArray[i] > 1)
                {
                    sources.Add(i);
                }
            }

            return sources;
        }

        /// <summary>
        /// Creates a broadcasted tensor by replicating data along broadcast dimensions.
        /// </summary>
        /// <param name="tensor">The tensor to broadcast.</param>
        /// <param name="originalShape">The original shape.</param>
        /// <param name="broadcastShape">The broadcast shape.</param>
        /// <returns>A new broadcasted tensor.</returns>
        private Tensor CreateBroadcastedTensor(Tensor tensor, int[] originalShape, int[] broadcastShape)
        {
            // Calculate total size of broadcasted tensor
            int totalSize = 1;
            foreach (int dim in broadcastShape)
            {
                totalSize *= dim;
            }

            float[] broadcastedData = new float[totalSize];

            // Pad original shape to match broadcast shape rank
            int rankDiff = broadcastShape.Length - originalShape.Length;
            int[] paddedOriginalShape = new int[broadcastShape.Length];
            for (int i = 0; i < rankDiff; i++)
            {
                paddedOriginalShape[i] = 1;
            }
            Array.Copy(originalShape, 0, paddedOriginalShape, rankDiff, originalShape.Length);

            // Broadcast by iterating through all indices
            for (int flatIndex = 0; flatIndex < totalSize; flatIndex++)
            {
                // Convert flat index to multi-dimensional indices in broadcast shape
                int[] broadcastIndices = FlatIndexToIndices(flatIndex, broadcastShape);

                // Map to original indices (using 0 for broadcast dimensions)
                int[] originalIndices = new int[broadcastShape.Length];
                for (int i = 0; i < broadcastShape.Length; i++)
                {
                    if (paddedOriginalShape[i] == 1 && broadcastShape[i] > 1)
                    {
                        // This dimension is broadcast, use index 0
                        originalIndices[i] = 0;
                    }
                    else
                    {
                        originalIndices[i] = broadcastIndices[i];
                    }
                }

                // Remove padding from original indices
                int[] unpaddedOriginalIndices = new int[originalShape.Length];
                Array.Copy(originalIndices, rankDiff, unpaddedOriginalIndices, 0, originalShape.Length);

                // Get value from original tensor
                int originalFlatIndex = IndicesToFlatIndex(unpaddedOriginalIndices, originalShape);
                broadcastedData[flatIndex] = tensor.Data[originalFlatIndex];
            }

            return new Tensor(broadcastedData, broadcastShape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Converts a flat index to multi-dimensional indices.
        /// </summary>
        /// <param name="flatIndex">The flat index.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>An array of indices.</returns>
        private int[] FlatIndexToIndices(int flatIndex, int[] shape)
        {
            int[] indices = new int[shape.Length];
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                indices[i] = flatIndex % shape[i];
                flatIndex /= shape[i];
            }
            return indices;
        }

        /// <summary>
        /// Converts multi-dimensional indices to a flat index.
        /// </summary>
        /// <param name="indices">The multi-dimensional indices.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The flat index.</returns>
        private int IndicesToFlatIndex(int[] indices, int[] shape)
        {
            int flatIndex = 0;
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                flatIndex += indices[i] * stride;
                stride *= shape[i];
            }
            return flatIndex;
        }

        /// <summary>
        /// Returns a string representation of this broadcasted tensor.
        /// </summary>
        /// <returns>A string describing the broadcast.</returns>
        public override string ToString()
        {
            return $"BroadcastedTensor: {OriginalShape} -> {BroadcastShape}";
        }
    }
}
