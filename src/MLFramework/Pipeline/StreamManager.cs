using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MLFramework.HAL;
using MLFramework.HAL.CUDA;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Manages CUDA streams for pipeline execution
    /// </summary>
    public class StreamManager : IDisposable
    {
        private readonly List<CudaStream> _streams;
        private readonly CudaDevice _device;
        private int _currentStreamIndex;
        private int _disposed;

        /// <summary>
        /// Number of managed streams
        /// </summary>
        public int Count => _streams.Count;

        /// <summary>
        /// Gets the device associated with this stream manager
        /// </summary>
        public CudaDevice Device => _device;

        public StreamManager(CudaDevice device, int numStreams)
        {
            _device = device ?? throw new ArgumentNullException(nameof(device));

            if (numStreams <= 0)
            {
                throw new ArgumentException("Number of streams must be greater than 0", nameof(numStreams));
            }

            _streams = new List<CudaStream>(numStreams);
            _currentStreamIndex = 0;

            // Create streams
            for (int i = 0; i < numStreams; i++)
            {
                _streams.Add(new CudaStream(device));
            }
        }

        /// <summary>
        /// Get a stream (round-robin)
        /// </summary>
        public CudaStream GetStream()
        {
            ThrowIfDisposed();

            lock (_streams)
            {
                int index = _currentStreamIndex;
                _currentStreamIndex = (_currentStreamIndex + 1) % _streams.Count;
                return _streams[index];
            }
        }

        /// <summary>
        /// Get a specific stream by index
        /// </summary>
        public CudaStream GetStream(int index)
        {
            ThrowIfDisposed();

            if (index < 0 || index >= _streams.Count)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(index),
                    $"Stream index must be in range [0, {_streams.Count - 1}]");
            }

            return _streams[index];
        }

        /// <summary>
        /// Synchronize all streams
        /// </summary>
        public Task SynchronizeAllAsync()
        {
            ThrowIfDisposed();

            return Task.Run(() =>
            {
                lock (_streams)
                {
                    foreach (var stream in _streams)
                    {
                        stream.Synchronize();
                    }
                }
            });
        }

        /// <summary>
        /// Synchronize a specific stream
        /// </summary>
        public Task SynchronizeAsync(int index)
        {
            ThrowIfDisposed();

            if (index < 0 || index >= _streams.Count)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(index),
                    $"Stream index must be in range [0, {_streams.Count - 1}]");
            }

            return Task.Run(() =>
            {
                var stream = _streams[index];
                stream.Synchronize();
            });
        }

        /// <summary>
        /// Record an event on a stream
        /// </summary>
        public CudaEvent RecordEvent(int streamIndex)
        {
            ThrowIfDisposed();

            if (streamIndex < 0 || streamIndex >= _streams.Count)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(streamIndex),
                    $"Stream index must be in range [0, {_streams.Count - 1}]");
            }

            var stream = _streams[streamIndex];
            return (CudaEvent)stream.RecordEvent();
        }

        /// <summary>
        /// Wait for an event on a stream
        /// </summary>
        public void WaitForEvent(CudaEvent evt, int streamIndex)
        {
            ThrowIfDisposed();

            if (evt == null)
                throw new ArgumentNullException(nameof(evt));

            if (streamIndex < 0 || streamIndex >= _streams.Count)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(streamIndex),
                    $"Stream index must be in range [0, {_streams.Count - 1}]");
            }

            var stream = _streams[streamIndex];
            stream.Wait(evt);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed == 1)
                throw new ObjectDisposedException(nameof(StreamManager));
        }

        public void Dispose()
        {
            if (_disposed == 1)
                return;

            lock (_streams)
            {
                foreach (var stream in _streams)
                {
                    stream.Dispose();
                }
                _streams.Clear();
            }

            _disposed = 1;
        }
    }
}
