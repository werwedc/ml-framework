using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Metadata about a micro-batch
    /// </summary>
    public class MicroBatchInfo
    {
        /// <summary>
        /// Index of this micro-batch (0 to numMicroBatches-1)
        /// </summary>
        public int Index { get; }

        /// <summary>
        /// Size of this micro-batch
        /// </summary>
        public int Size { get; }

        /// <summary>
        /// Start index in the original batch
        /// </summary>
        public int StartIndex { get; }

        /// <summary>
        /// End index (exclusive) in the original batch
        /// </summary>
        public int EndIndex { get; }

        public MicroBatchInfo(int index, int size, int startIndex, int endIndex)
        {
            if (index < 0)
                throw new ArgumentOutOfRangeException(nameof(index), "Index must be non-negative");
            if (size <= 0)
                throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive");
            if (startIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(startIndex), "Start index must be non-negative");
            if (endIndex <= startIndex)
                throw new ArgumentException("End index must be greater than start index", nameof(endIndex));

            Index = index;
            Size = size;
            StartIndex = startIndex;
            EndIndex = endIndex;
        }
    }

    /// <summary>
    /// Manages micro-batching and gradient accumulation for pipeline parallelism
    /// </summary>
    public class MicroBatchManager : IDisposable
    {
        private readonly int _microBatchSize;
        private readonly int _numMicroBatches;
        private readonly int _totalBatchSize;
        private readonly IDevice _device;
        private readonly List<Tensor?> _accumulatedGradients;
        private int _currentMicroBatch;
        private int _disposed;

        /// <summary>
        /// Size of each micro-batch
        /// </summary>
        public int MicroBatchSize => _microBatchSize;

        /// <summary>
        /// Number of micro-batches per full batch
        /// </summary>
        public int NumMicroBatches => _numMicroBatches;

        /// <summary>
        /// Total batch size (microBatchSize * numMicroBatches)
        /// </summary>
        public int TotalBatchSize => _totalBatchSize;

        /// <summary>
        /// Check if all micro-batches have been processed
        /// </summary>
        public bool IsComplete => _currentMicroBatch >= _numMicroBatches;

        public MicroBatchManager(int totalBatchSize, int numMicroBatches, IDevice device)
        {
            if (totalBatchSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(totalBatchSize), "Total batch size must be positive");
            if (numMicroBatches <= 0)
                throw new ArgumentOutOfRangeException(nameof(numMicroBatches), "Number of micro-batches must be positive");
            if (device == null)
                throw new ArgumentNullException(nameof(device));

            _totalBatchSize = totalBatchSize;
            _numMicroBatches = numMicroBatches;
            _microBatchSize = totalBatchSize / numMicroBatches;
            _device = device;
            _accumulatedGradients = new List<Tensor?>();
            _currentMicroBatch = 0;
        }

        /// <summary>
        /// Split a batch into micro-batches
        /// </summary>
        /// <returns>List of micro-batch tensors</returns>
        public List<Tensor> SplitBatch(Tensor batch)
        {
            ThrowIfDisposed();

            if (batch == null)
                throw new ArgumentNullException(nameof(batch));
            if (batch.Shape.Length == 0)
                throw new ArgumentException("Batch tensor must have at least 1 dimension", nameof(batch));

            if (batch.Shape[0] != _totalBatchSize)
                throw new ArgumentException($"Batch dimension {batch.Shape[0]} does not match total batch size {_totalBatchSize}", nameof(batch));

            var microBatches = new List<Tensor>(_numMicroBatches);
            int startIndex = 0;

            for (int i = 0; i < _numMicroBatches; i++)
            {
                int microBatchSize = _microBatchSize;

                // Last micro-batch gets the remainder
                if (i == _numMicroBatches - 1)
                {
                    microBatchSize = _totalBatchSize - startIndex;
                }

                // Slice the batch
                var microBatch = SliceTensor(batch, startIndex, startIndex + microBatchSize);
                microBatches.Add(microBatch);
                startIndex += microBatchSize;
            }

            return microBatches;
        }

        /// <summary>
        /// Combine micro-batch outputs into a single batch
        /// </summary>
        public Tensor CombineOutputs(List<Tensor> microBatchOutputs)
        {
            ThrowIfDisposed();

            if (microBatchOutputs == null)
                throw new ArgumentNullException(nameof(microBatchOutputs));
            if (microBatchOutputs.Count != _numMicroBatches)
                throw new ArgumentException($"Expected {_numMicroBatches} outputs, got {microBatchOutputs.Count}", nameof(microBatchOutputs));

            // Calculate total batch size from micro-batches
            int totalSize = microBatchOutputs.Sum(m => (int)m.Shape[0]);

            // Create combined tensor
            var firstOutput = microBatchOutputs[0];
            var combinedShape = new int[firstOutput.Shape.Length];
            combinedShape[0] = (int)totalSize;
            Array.Copy(firstOutput.Shape, 1, combinedShape, 1, firstOutput.Shape.Length - 1);

            var combined = Tensor.Zeros(combinedShape);
            int startIndex = 0;

            foreach (var microBatch in microBatchOutputs)
            {
                // Copy micro-batch data to combined tensor
                int microBatchSize = (int)microBatch.Shape[0];
                int dataSize = microBatch.Data.Length;
                Array.Copy(microBatch.Data, 0, combined.Data, startIndex, dataSize);
                startIndex += dataSize;
            }

            return combined;
        }

        /// <summary>
        /// Accumulate gradients from a micro-batch
        /// </summary>
        public void AccumulateGradients(IEnumerable<Tensor> gradients)
        {
            ThrowIfDisposed();

            if (gradients == null)
                throw new ArgumentNullException(nameof(gradients));

            var gradientList = gradients.ToList();

            // Initialize accumulated gradients on first micro-batch
            if (_accumulatedGradients.Count == 0)
            {
                foreach (var grad in gradientList)
                {
                    if (grad == null)
                        throw new ArgumentNullException(nameof(gradients), "Gradient tensor cannot be null");
                    _accumulatedGradients.Add(grad.Clone());
                }
            }
            else
            {
                // Add gradients to accumulated ones
                if (gradientList.Count != _accumulatedGradients.Count)
                {
                    throw new ArgumentException(
                        $"Expected {_accumulatedGradients.Count} gradients, got {gradientList.Count}",
                        nameof(gradients));
                }

                for (int i = 0; i < gradientList.Count; i++)
                {
                    if (_accumulatedGradients[i] == null || gradientList[i] == null)
                        throw new InvalidOperationException("Gradient tensor is null");

                    if (!_accumulatedGradients[i].Shape.SequenceEqual(gradientList[i].Shape))
                    {
                        throw new ArgumentException(
                            $"Gradient {i} shape mismatch: {_accumulatedGradients[i].Shape} vs {gradientList[i].Shape}");
                    }

                    // Add gradients
                    for (int j = 0; j < _accumulatedGradients[i]!.Data.Length; j++)
                    {
                        _accumulatedGradients[i]!.Data[j] += gradientList[i].Data[j];
                    }
                }
            }

            _currentMicroBatch++;
        }

        /// <summary>
        /// Get the accumulated gradients (averaged over micro-batches)
        /// </summary>
        public List<Tensor> GetAccumulatedGradients()
        {
            ThrowIfDisposed();

            if (!IsComplete && _accumulatedGradients.Count > 0)
            {
                // Warning but still return what we have
                Console.WriteLine($"Warning: Not all micro-batches processed ({_currentMicroBatch}/{_numMicroBatches})");
            }

            // Average gradients
            var averagedGradients = new List<Tensor>();
            foreach (var grad in _accumulatedGradients)
            {
                if (grad == null)
                {
                    averagedGradients.Add(null!);
                    continue;
                }

                var averaged = grad.Clone();
                float scale = 1.0f / _numMicroBatches;
                for (int i = 0; i < averaged.Data.Length; i++)
                {
                    averaged.Data[i] *= scale;
                }
                averagedGradients.Add(averaged);
            }

            return averagedGradients;
        }

        /// <summary>
        /// Reset accumulated gradients to zero
        /// </summary>
        public void ResetGradients()
        {
            ThrowIfDisposed();

            _currentMicroBatch = 0;

            // Zero out accumulated gradients
            foreach (var grad in _accumulatedGradients)
            {
                if (grad != null)
                {
                    Array.Clear(grad.Data, 0, grad.Data.Length);
                }
            }
        }

        private Tensor SliceTensor(Tensor tensor, int start, int end)
        {
            // Get dimensions
            int[] fullShape = tensor.Shape;
            long batchSize = end - start;

            // Calculate size of remaining dimensions
            long remainingSize = 1;
            for (int i = 1; i < fullShape.Length; i++)
            {
                remainingSize *= fullShape[i];
            }

            // Create sliced tensor - return int[] shape for Tensor constructor
            var intShape = new int[fullShape.Length];
            intShape[0] = (int)batchSize;
            Array.Copy(fullShape, 1, intShape, 1, fullShape.Length - 1);

            // Create sliced tensor with correct array type
            var slice = new Tensor(new float[batchSize * remainingSize], intShape);

            // Copy data
            int startIdx = (int)(start * remainingSize);
            Array.Copy(tensor.Data, startIdx, slice.Data, 0, slice.Data.Length);

            return slice;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed == 1)
                throw new ObjectDisposedException(nameof(MicroBatchManager));
        }

        public void Dispose()
        {
            if (_disposed == 1)
                return;

            foreach (var grad in _accumulatedGradients)
            {
                // Let the GC handle tensor disposal
            }

            _accumulatedGradients.Clear();
            _disposed = 1;
        }
    }
}
