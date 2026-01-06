using System;
using System.Collections;
using System.Collections.Generic;

namespace MLFramework.Data;

/// <summary>
/// Replicates a single stream across multiple workers, ensuring each worker gets disjoint samples.
/// Uses stride-based partitioning to distribute samples across replicas.
/// </summary>
/// <typeparam name="T">The type of data items in the stream.</typeparam>
public class StreamReplicator<T>
{
    private readonly IEnumerator<T> _sourceStream;
    private readonly int _numReplicas;

    /// <summary>
    /// Initializes a new instance of the StreamReplicator class.
    /// </summary>
    /// <param name="sourceStream">The source stream to replicate.</param>
    /// <param name="numReplicas">The number of replicas/workers.</param>
    public StreamReplicator(IEnumerator<T> sourceStream, int numReplicas)
    {
        _sourceStream = sourceStream ?? throw new ArgumentNullException(nameof(sourceStream));

        if (numReplicas < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numReplicas), "Number of replicas must be at least 1.");
        }

        _numReplicas = numReplicas;
    }

    /// <summary>
    /// Gets a replica stream for a specific replica ID.
    /// Each replica receives a subset of samples from the source stream.
    /// </summary>
    /// <param name="replicaId">The ID of the replica (0-based).</param>
    /// <returns>An enumerator that provides samples for this replica.</returns>
    public IEnumerator<T> GetReplicaStream(int replicaId)
    {
        if (replicaId < 0 || replicaId >= _numReplicas)
        {
            throw new ArgumentOutOfRangeException(
                nameof(replicaId),
                $"Replica ID must be between 0 and {_numReplicas - 1}.");
        }

        return new ReplicaEnumerator(_sourceStream, replicaId, _numReplicas);
    }

    /// <summary>
    /// Inner enumerator class that provides stride-based partitioning of the source stream.
    /// Each replica reads samples at positions where (position % numReplicas == replicaId).
    /// </summary>
    private class ReplicaEnumerator : IEnumerator<T>
    {
        private readonly IEnumerator<T> _source;
        private readonly int _replicaId;
        private readonly int _stride;
        private int _position;

        /// <summary>
        /// Initializes a new instance of the ReplicaEnumerator class.
        /// </summary>
        /// <param name="source">The source enumerator.</param>
        /// <param name="replicaId">The ID of this replica.</param>
        /// <param name="stride">The stride for partitioning.</param>
        public ReplicaEnumerator(IEnumerator<T> source, int replicaId, int stride)
        {
            _source = source ?? throw new ArgumentNullException(nameof(source));
            _replicaId = replicaId;
            _stride = stride;
            _position = -1;
        }

        /// <summary>
        /// Gets the element in the collection at the current position of the enumerator.
        /// </summary>
        public T Current => _source.Current;

        /// <summary>
        /// Gets the element in the collection at the current position of the enumerator.
        /// </summary>
        object IEnumerator.Current => Current!;

        /// <summary>
        /// Advances the enumerator to the next element of the collection.
        /// </summary>
        /// <returns>true if the enumerator was successfully advanced to the next element; false if the enumerator has passed the end of the collection.</returns>
        public bool MoveNext()
        {
            _position++;

            // Move source to find the next sample for this replica
            while (_source.MoveNext())
            {
                // Check if this sample belongs to this replica
                if (_position % _stride == _replicaId)
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Sets the enumerator to its initial position, which is before the first element in the collection.
        /// This is not supported for streaming data.
        /// </summary>
        /// <exception cref="NotSupportedException">Always thrown, as streaming data cannot be reset.</exception>
        public void Reset()
        {
            throw new NotSupportedException("Reset is not supported for streaming data.");
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// Note: This does not dispose the shared source stream, as it may be used by other replicas.
        /// </summary>
        public void Dispose()
        {
            // Don't dispose the shared source stream, as it may be used by other replicas
            GC.SuppressFinalize(this);
        }
    }
}
