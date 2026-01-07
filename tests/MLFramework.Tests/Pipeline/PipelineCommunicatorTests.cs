using System;
using System.Threading.Tasks;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Pipeline;

namespace MLFramework.Tests.Pipeline
{
    public class PipelineCommunicatorTests : IDisposable
    {
        private LocalPipelineCommunicator[]? _communicators;

        public void Dispose()
        {
            if (_communicators != null)
            {
                foreach (var comm in _communicators)
                {
                    comm?.Dispose();
                }
                _communicators = null;
            }
        }

        private LocalPipelineCommunicator[] CreateCommunicators(int worldSize)
        {
            var communicators = new LocalPipelineCommunicator[worldSize];
            for (int i = 0; i < worldSize; i++)
            {
                communicators[i] = new LocalPipelineCommunicator(i, worldSize);
            }
            _communicators = communicators;
            return communicators;
        }

        private Tensor CreateTestTensor(int size)
        {
            var data = new float[size];
            for (int i = 0; i < size; i++)
            {
                data[i] = (float)i;
            }
            return new Tensor(data, new long[] { size });
        }

        [Fact]
        public async Task SendAndReceive_BetweenTwoRanks_WorksCorrectly()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var sender = communicators[0];
            var receiver = communicators[1];
            var expectedTensor = CreateTestTensor(10);

            // Act
            var sendTask = sender.SendAsync(expectedTensor, 1);
            var receiveTask = receiver.ReceiveAsync(0);

            await Task.WhenAll(sendTask, receiveTask);
            var receivedTensor = receiveTask.Result;

            // Assert
            Assert.NotNull(receivedTensor);
            Assert.Equal(expectedTensor.Shape, receivedTensor.Shape);
            Assert.Equal(expectedTensor.Data, receivedTensor.Data);
        }

        [Fact]
        public async Task SendAndReceive_WithLargeTensor_WorksCorrectly()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var sender = communicators[0];
            var receiver = communicators[1];
            var expectedTensor = CreateTestTensor(1000);

            // Act
            var sendTask = sender.SendAsync(expectedTensor, 1);
            var receiveTask = receiver.ReceiveAsync(0);

            await Task.WhenAll(sendTask, receiveTask);
            var receivedTensor = receiveTask.Result;

            // Assert
            Assert.NotNull(receivedTensor);
            Assert.Equal(expectedTensor.Data.Length, receivedTensor.Data.Length);
        }

        [Fact]
        public async Task MultipleSequentialSends_WorksCorrectly()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var sender = communicators[0];
            var receiver = communicators[1];
            int numSends = 5;

            // Act
            for (int i = 0; i < numSends; i++)
            {
                var tensor = CreateTestTensor(10 * (i + 1));
                await sender.SendAsync(tensor, 1);
                var received = await receiver.ReceiveAsync(0);

                Assert.Equal(tensor.Shape, received.Shape);
            }

            // Assert - if we got here without exceptions, test passed
        }

        [Fact]
        public async Task Barrier_SynchronizesProcesses()
        {
            // Arrange
            var communicators = CreateCommunicators(3);
            var tasks = new Task[3];

            // Act - All ranks call barrier
            for (int i = 0; i < 3; i++)
            {
                int rank = i;
                tasks[i] = Task.Run(async () =>
                {
                    await Task.Delay(rank * 100); // Stagger execution
                    await communicators[rank].BarrierAsync();
                });
            }

            await Task.WhenAll(tasks);

            // Assert - if all tasks completed, barrier worked
        }

        [Fact]
        public async Task Broadcast_FromRootToAll_DistributesCorrectly()
        {
            // Arrange
            var communicators = CreateCommunicators(4);
            var rootTensor = CreateTestTensor(10);
            var tasks = new Task<Tensor>[4];

            // Act - All ranks participate in broadcast
            for (int i = 0; i < 4; i++)
            {
                int rank = i;
                tasks[i] = Task.Run(async () =>
                {
                    return await communicators[rank].BroadcastAsync(rootTensor, root: 0);
                });
            }

            var results = await Task.WhenAll(tasks);

            // Assert
            foreach (var result in results)
            {
                Assert.NotNull(result);
                Assert.Equal(rootTensor.Shape, result.Shape);
                Assert.Equal(rootTensor.Data, result.Data);
            }
        }

        [Fact]
        public async Task Send_WithInvalidDestination_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var sender = communicators[0];
            var tensor = CreateTestTensor(10);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
                sender.SendAsync(tensor, 2)); // Invalid rank
        }

        [Fact]
        public async Task Send_ToSelf_ThrowsArgumentException()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var sender = communicators[0];
            var tensor = CreateTestTensor(10);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() =>
                sender.SendAsync(tensor, 0)); // Sending to self
        }

        [Fact]
        public async Task Receive_WithInvalidSource_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var receiver = communicators[0];

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
                receiver.ReceiveAsync(2)); // Invalid rank
        }

        [Fact]
        public async Task Receive_FromSelf_ThrowsArgumentException()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var receiver = communicators[0];

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() =>
                receiver.ReceiveAsync(0)); // Receiving from self
        }

        [Fact]
        public async Task Send_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var sender = communicators[0];
            var tensor = CreateTestTensor(10);
            sender.Dispose();

            // Act & Assert
            await Assert.ThrowsAsync<ObjectDisposedException>(() =>
                sender.SendAsync(tensor, 1));
        }

        [Fact]
        public async Task Receive_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var receiver = communicators[1];
            receiver.Dispose();

            // Act & Assert
            await Assert.ThrowsAsync<ObjectDisposedException>(() =>
                receiver.ReceiveAsync(0));
        }

        [Fact]
        public async Task Barrier_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var communicator = communicators[0];
            communicator.Dispose();

            // Act & Assert
            await Assert.ThrowsAsync<ObjectDisposedException>(() =>
                communicator.BarrierAsync());
        }

        [Fact]
        public void Constructor_WithValidInputs_CreatesCommunicator()
        {
            // Act
            var communicator = new LocalPipelineCommunicator(rank: 0, worldSize: 4);

            // Assert
            Assert.Equal(0, communicator.Rank);
            Assert.Equal(4, communicator.WorldSize);
        }

        [Fact]
        public void Constructor_WithInvalidRank_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new LocalPipelineCommunicator(rank: -1, worldSize: 4));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new LocalPipelineCommunicator(rank: 4, worldSize: 4));
        }

        [Fact]
        public void Constructor_WithInvalidWorldSize_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new LocalPipelineCommunicator(rank: 0, worldSize: 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new LocalPipelineCommunicator(rank: 0, worldSize: -1));
        }

        [Fact]
        public async Task Send_WithNullTensor_ThrowsArgumentNullException()
        {
            // Arrange
            var communicators = CreateCommunicators(2);
            var sender = communicators[0];

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                sender.SendAsync(null!, 1));
        }

        [Fact]
        public async Task ForwardCommunication_ThroughMultipleStages_WorksCorrectly()
        {
            // Arrange
            var communicators = CreateCommunicators(4);
            var tensor = CreateTestTensor(10);
            var tasks = new Task[4];

            // Act - Simulate forward communication through pipeline
            // Rank 0 sends to 1, 1 sends to 2, 2 sends to 3
            for (int i = 0; i < 4; i++)
            {
                int rank = i;
                tasks[i] = Task.Run(async () =>
                {
                    if (rank == 0)
                    {
                        await communicators[rank].SendAsync(tensor, 1);
                    }
                    else if (rank < 3)
                    {
                        var received = await communicators[rank].ReceiveAsync(rank - 1);
                        await communicators[rank].SendAsync(received, rank + 1);
                    }
                    else // rank == 3
                    {
                        var received = await communicators[rank].ReceiveAsync(rank - 1);
                        return received;
                    }
                    return null;
                });
            }

            var results = await Task.WhenAll(tasks);

            // Assert
            var finalTensor = results[3];
            Assert.NotNull(finalTensor);
            Assert.Equal(tensor.Data, finalTensor.Data);
        }

        [Fact]
        public async Task BackwardCommunication_ThroughMultipleStages_WorksCorrectly()
        {
            // Arrange
            var communicators = CreateCommunicators(4);
            var gradient = CreateTestTensor(10);
            var tasks = new Task[4];

            // Act - Simulate backward communication through pipeline
            // Rank 3 sends to 2, 2 sends to 1, 1 sends to 0
            for (int i = 0; i < 4; i++)
            {
                int rank = i;
                tasks[i] = Task.Run(async () =>
                {
                    if (rank == 3)
                    {
                        await communicators[rank].SendAsync(gradient, 2);
                    }
                    else if (rank > 0)
                    {
                        var received = await communicators[rank].ReceiveAsync(rank + 1);
                        await communicators[rank].SendAsync(received, rank - 1);
                    }
                    else // rank == 0
                    {
                        var received = await communicators[rank].ReceiveAsync(1);
                        return received;
                    }
                    return null;
                });
            }

            var results = await Task.WhenAll(tasks);

            // Assert
            var finalGradient = results[0];
            Assert.NotNull(finalGradient);
            Assert.Equal(gradient.Data, finalGradient.Data);
        }
    }
}
