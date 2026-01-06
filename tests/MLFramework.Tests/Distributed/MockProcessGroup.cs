using MLFramework.Distributed;
using MLFramework.Tensor;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Tests.Distributed
{
    /// <summary>
    /// Mock process group for testing without actual distributed hardware.
    /// </summary>
    public class MockProcessGroup : IProcessGroup
    {
        private static MockProcessGroup _currentGroup;
        private static readonly object _lock = new object();

        private readonly int _rank;
        private readonly int _worldSize;
        private readonly Dictionary<int, List<Tensor>> _mockState;
        private readonly MockBackend _backend;

        public MockProcessGroup(int rank, int worldSize)
        {
            _rank = rank;
            _worldSize = worldSize;
            _mockState = new Dictionary<int, List<Tensor>>();
            _backend = new MockBackend();

            for (int i = 0; i < worldSize; i++)
            {
                _mockState[i] = new List<Tensor>();
            }
        }

        public int Rank => _rank;
        public int WorldSize => _worldSize;
        public ICommunicationBackend Backend => _backend;

        public static MockProcessGroup Create(int worldSize, int rank)
        {
            lock (_lock)
            {
                if (_currentGroup != null)
                {
                    throw new InvalidOperationException("Only one active process group is allowed");
                }

                _currentGroup = new MockProcessGroup(rank, worldSize);
                return _currentGroup;
            }
        }

        public void AllReduce(Tensor tensor, ReduceOp op)
        {
            // Store tensor for this rank
            _mockState[_rank].Add(tensor.Clone());

            // Simulate reduction by summing all tensors (simplified)
            var allTensors = new List<Tensor>();
            for (int i = 0; i < _worldSize; i++)
            {
                if (_mockState[i].Count > _rank)
                {
                    allTensors.Add(_mockState[i][_rank]);
                }
            }

            if (allTensors.Count == _worldSize)
            {
                // All ranks have provided their tensor, perform reduction
                var sum = allTensors[0].Clone();
                for (int i = 1; i < allTensors.Count; i++)
                {
                    sum.Add_(allTensors[i]);
                }

                if (op == ReduceOp.Avg)
                {
                    sum.Div_(_worldSize);
                }
                else if (op == ReduceOp.Max)
                {
                    var max = allTensors[0].Clone();
                    for (int i = 1; i < allTensors.Count; i++)
                    {
                        max = Tensor.Maximum(max, allTensors[i]);
                    }
                    sum.Copy_(max);
                }

                // All ranks get the reduced result
                tensor.Copy_(sum);
            }
        }

        public void Broadcast(Tensor tensor, int root)
        {
            // Simplified: copy tensor from root to all ranks
            // In tests, we can just use shared state
            if (_rank == root)
            {
                _mockState[root].Add(tensor.Clone());
            }
            else
            {
                if (_mockState[root].Count > 0)
                {
                    tensor.Copy_(_mockState[root][0]);
                }
            }
        }

        public void Barrier()
        {
            // No-op in mock
        }

        public void Scatter(Tensor output, Tensor input, int root)
        {
            // Simplified mock implementation
            if (_rank == root)
            {
                var chunkSize = input.Numel / _worldSize;
                var start = _rank * chunkSize;
                var end = start + chunkSize;
                var chunk = input.Slice(0, start, end - start);
                output.Copy_(chunk);
            }
        }

        public void Gather(Tensor output, Tensor input, int root)
        {
            // Simplified mock implementation
            if (_rank == root)
            {
                var chunkSize = input.Numel / _worldSize;
                output.Copy_(input);
            }
        }

        public void AllGather(Tensor output, Tensor input)
        {
            // Simplified mock implementation
            output.Copy_(input);
        }

        public void Reduce(Tensor output, Tensor input, ReduceOp op, int root)
        {
            // Simplified mock implementation
            output.Copy_(input);
            if (op == ReduceOp.Avg && _worldSize > 1)
            {
                output.Div_(_worldSize);
            }
        }

        // Async versions just wrap sync versions
        public Task AllReduceAsync(Tensor tensor, ReduceOp op)
        {
            AllReduce(tensor, op);
            return Task.CompletedTask;
        }

        public Task BroadcastAsync(Tensor tensor, int root)
        {
            Broadcast(tensor, root);
            return Task.CompletedTask;
        }

        public Task BarrierAsync()
        {
            Barrier();
            return Task.CompletedTask;
        }

        public void Send(Tensor tensor, int dst)
        {
            // Simplified mock implementation
            _mockState[dst].Add(tensor.Clone());
        }

        public void Recv(Tensor tensor, int src)
        {
            // Simplified mock implementation
            if (_mockState[src].Count > 0)
            {
                tensor.Copy_(_mockState[src][0]);
            }
        }

        public Task SendAsync(Tensor tensor, int dst)
        {
            Send(tensor, dst);
            return Task.CompletedTask;
        }

        public Task RecvAsync(Tensor tensor, int src)
        {
            Recv(tensor, src);
            return Task.CompletedTask;
        }

        public Task ScatterAsync(Tensor output, Tensor input, int root)
        {
            Scatter(output, input, root);
            return Task.CompletedTask;
        }

        public Task GatherAsync(Tensor output, Tensor input, int root)
        {
            Gather(output, input, root);
            return Task.CompletedTask;
        }

        public Task AllGatherAsync(Tensor output, Tensor input)
        {
            AllGather(output, input);
            return Task.CompletedTask;
        }

        public Task ReduceAsync(Tensor output, Tensor input, ReduceOp op, int root)
        {
            Reduce(output, input, op, root);
            return Task.CompletedTask;
        }

        public void Destroy()
        {
            lock (_lock)
            {
                if (_currentGroup == this)
                {
                    _currentGroup = null;
                }
            }
        }

        /// <summary>
        /// Helper to simulate all ranks providing tensors.
        /// </summary>
        public void SimulateAllReduce(List<Tensor> tensors, ReduceOp op)
        {
            var sum = tensors[0].Clone();
            for (int i = 1; i < tensors.Count; i++)
            {
                sum.Add_(tensors[i]);
            }

            if (op == ReduceOp.Avg)
            {
                sum.Div_(tensors.Count);
            }
            else if (op == ReduceOp.Max)
            {
                var max = tensors[0].Clone();
                for (int i = 1; i < tensors.Count; i++)
                {
                    max = Tensor.Maximum(max, tensors[i]);
                }
                sum.Copy_(max);
            }

            // Return reduced tensor to all ranks
            foreach (var tensor in tensors)
            {
                tensor.Copy_(sum);
            }
        }
    }

    /// <summary>
    /// Mock communication backend for testing.
    /// </summary>
    public class MockBackend : ICommunicationBackend
    {
        public string Name => "MockBackend";
        public bool IsAvailable => true;
        public int DeviceCount => 4;
        public bool SupportsAsync => true;
        public bool SupportsGPUDirect => false;
        public long GetBufferSizeLimit() => 1024 * 1024 * 1024; // 1GB
    }


