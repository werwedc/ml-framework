using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Manages gradient accumulation for micro-batching in FSDP.
    /// Accumulates gradients from multiple micro-batches before optimizer step.
    /// </summary>
    public class GradientAccumulator : IDisposable
    {
        private readonly Dictionary<string, List<Tensor>> _accumulatedGradients;
        private readonly int _accumulationSteps;
        private bool _disposed;

        /// <summary>
        /// Initialize a gradient accumulator.
        /// </summary>
        /// <param name="accumulationSteps">Number of micro-batches to accumulate</param>
        /// <exception cref="ArgumentException">Thrown when accumulation steps is not positive</exception>
        public GradientAccumulator(int accumulationSteps = 1)
        {
            if (accumulationSteps <= 0)
                throw new ArgumentException("Accumulation steps must be positive", nameof(accumulationSteps));

            _accumulatedGradients = new Dictionary<string, List<Tensor>>();
            _accumulationSteps = accumulationSteps;
        }

        /// <summary>
        /// Get the number of accumulation steps.
        /// </summary>
        public int AccumulationSteps => _accumulationSteps;

        /// <summary>
        /// Add gradients from a micro-batch.
        /// </summary>
        /// <param name="gradients">Gradients to add</param>
        /// <exception cref="ArgumentNullException">Thrown when gradients is null</exception>
        public void AddGradients(Dictionary<string, Tensor> gradients)
        {
            if (gradients == null)
                throw new ArgumentNullException(nameof(gradients));

            if (gradients.Count == 0)
                return;

            foreach (var kvp in gradients)
            {
                var paramName = kvp.Key;
                var grad = kvp.Value;

                if (!_accumulatedGradients.ContainsKey(paramName))
                {
                    _accumulatedGradients[paramName] = new List<Tensor>();
                }

                _accumulatedGradients[paramName].Add(grad.Clone());
            }
        }

        /// <summary>
        /// Get accumulated gradients and clear accumulator.
        /// </summary>
        /// <returns>Accumulated gradients</returns>
        public Dictionary<string, Tensor> GetAndClearGradients()
        {
            var result = new Dictionary<string, Tensor>();

            foreach (var kvp in _accumulatedGradients)
            {
                var paramName = kvp.Key;
                var grads = kvp.Value;

                if (grads.Count == 0)
                    continue;

                // Sum all accumulated gradients
                var summedGrad = grads[0].Clone();
                for (int i = 1; i < grads.Count; i++)
                {
                    var gradData = grads[i].Data;
                    for (int j = 0; j < gradData.Length; j++)
                    {
                        summedGrad.Data[j] += gradData[j];
                    }
                }

                result[paramName] = summedGrad;

                // Clear accumulated gradients
                foreach (var grad in grads)
                {
                    // Tensor doesn't implement IDisposable, so no dispose needed
                }
                grads.Clear();
            }

            return result;
        }

        /// <summary>
        /// Get current accumulated gradients without clearing.
        /// </summary>
        /// <returns>Accumulated gradients</returns>
        public Dictionary<string, Tensor> GetGradients()
        {
            var result = new Dictionary<string, Tensor>();

            foreach (var kvp in _accumulatedGradients)
            {
                var paramName = kvp.Key;
                var grads = kvp.Value;

                if (grads.Count == 0)
                    continue;

                // Sum all accumulated gradients
                var summedGrad = grads[0].Clone();
                for (int i = 1; i < grads.Count; i++)
                {
                    var gradData = grads[i].Data;
                    for (int j = 0; j < gradData.Length; j++)
                    {
                        summedGrad.Data[j] += gradData[j];
                    }
                }

                result[paramName] = summedGrad;
            }

            return result;
        }

        /// <summary>
        /// Check if accumulation is complete.
        /// </summary>
        public bool IsComplete
        {
            get
            {
                if (_accumulatedGradients.Count == 0)
                    return false;

                return _accumulatedGradients.Values.All(list => list.Count >= _accumulationSteps);
            }
        }

        /// <summary>
        /// Get the number of accumulated steps for a specific parameter.
        /// </summary>
        /// <param name="paramName">Parameter name</param>
        /// <returns>Number of accumulated steps</returns>
        public int GetAccumulatedCount(string paramName)
        {
            if (_accumulatedGradients.TryGetValue(paramName, out var grads))
            {
                return grads.Count;
            }
            return 0;
        }

        /// <summary>
        /// Get the total number of accumulated steps across all parameters.
        /// </summary>
        /// <returns>Total accumulated steps</returns>
        public int GetTotalAccumulatedCount()
        {
            if (_accumulatedGradients.Count == 0)
                return 0;

            return _accumulatedGradients.Values.Sum(list => list.Count);
        }

        /// <summary>
        /// Reset the accumulator.
        /// </summary>
        public void Reset()
        {
            foreach (var list in _accumulatedGradients.Values)
            {
                foreach (var grad in list)
                {
                    // Tensor doesn't implement IDisposable, so no dispose needed
                }
                list.Clear();
            }
            _accumulatedGradients.Clear();
        }

        /// <summary>
        /// Get the number of parameters being tracked.
        /// </summary>
        /// <returns>Number of tracked parameters</returns>
        public int GetTrackedParameterCount()
        {
            return _accumulatedGradients.Count;
        }

        /// <summary>
        /// Get the names of all parameters being tracked.
        /// </summary>
        /// <returns>List of parameter names</returns>
        public IEnumerable<string> GetTrackedParameterNames()
        {
            return _accumulatedGradients.Keys;
        }

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Protected implementation of dispose pattern.
        /// </summary>
        /// <param name="disposing">Whether managed resources should be disposed</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    Reset();
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for GradientAccumulator.
        /// </summary>
        ~GradientAccumulator()
        {
            Dispose(false);
        }
    }
}
