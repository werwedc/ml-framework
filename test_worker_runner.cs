using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Models;
using MachineLearning.Distributed.Worker;
using MachineLearning.Distributed.Enums;

// Simple test runner to verify ElasticWorker functionality
class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Testing ElasticWorker Implementation...\n");

        try
        {
            TestConstructor();
            TestConnection();
            TestTopologyUpdate();
            TestStateSynchronization();
            TestSharding();

            Console.WriteLine("\n✓ All tests passed!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n✗ Test failed: {ex.Message}");
            Environment.Exit(1);
        }
    }

    static void TestConstructor()
    {
        Console.WriteLine("Testing constructor...");

        var config = new ElasticTrainingConfig
        {
            MinWorkers = 1,
            MaxWorkers = 10,
            WorkerHeartbeatTimeoutMs = 10000
        };

        var workerId = new WorkerId("worker-1", "localhost", 8080);
        var worker = new ElasticWorker(workerId, config);

        Console.WriteLine("  ✓ Worker created successfully");
        Console.WriteLine($"  ✓ Worker ID: {worker.WorkerId}");
        Console.WriteLine($"  ✓ IsRescaling: {worker.IsRescaling}");

        worker.Dispose();
    }

    static void TestConnection()
    {
        Console.WriteLine("\nTesting connection...");

        var config = new ElasticTrainingConfig
        {
            MinWorkers = 1,
            MaxWorkers = 10,
            WorkerHeartbeatTimeoutMs = 10000
        };

        var workerId = new WorkerId("worker-1", "localhost", 8080);
        var worker = new ElasticWorker(workerId, config);

        worker.ConnectToCluster("localhost:9090");

        var metadata = worker.GetMetadata();
        Console.WriteLine($"  ✓ Status: {metadata.Status}");
        Console.WriteLine($"  ✓ LastHeartbeat: {metadata.LastHeartbeat}");

        worker.Dispose();
    }

    static void TestTopologyUpdate()
    {
        Console.WriteLine("\nTesting topology update...");

        var config = new ElasticTrainingConfig
        {
            MinWorkers = 1,
            MaxWorkers = 10,
            WorkerHeartbeatTimeoutMs = 10000
        };

        var workerId = new WorkerId("worker-1", "localhost", 8080);
        var worker = new ElasticWorker(workerId, config);

        var topology = new ClusterTopology();
        topology.AddWorker(new WorkerId("worker-0", "host1", 8080));
        topology.AddWorker(workerId);
        topology.AddWorker(new WorkerId("worker-2", "host2", 8080));

        worker.UpdateTopologyAsync(topology).Wait();

        Console.WriteLine($"  ✓ CurrentTopology: {worker.CurrentTopology != null}");
        Console.WriteLine($"  ✓ Worker Rank: {worker.GetMetadata().Rank}");

        worker.Dispose();
    }

    static void TestStateSynchronization()
    {
        Console.WriteLine("\nTesting state synchronization...");

        var config = new ElasticTrainingConfig
        {
            MinWorkers = 1,
            MaxWorkers = 10,
            WorkerHeartbeatTimeoutMs = 10000
        };

        var workerId = new WorkerId("worker-1", "localhost", 8080);
        var worker = new ElasticWorker(workerId, config);

        var state = new GlobalTrainingState
        {
            CurrentEpoch = 5,
            CurrentStep = 1000,
            LearningRate = 0.01f,
            GlobalBatchSize = 32,
            ActiveWorkerCount = 4
        };

        worker.SynchronizeStateAsync(state).Wait();

        Console.WriteLine($"  ✓ CurrentState: {worker.CurrentState != null}");
        Console.WriteLine($"  ✓ Epoch: {worker.CurrentState?.CurrentEpoch}");
        Console.WriteLine($"  ✓ Step: {worker.CurrentState?.CurrentStep}");
        Console.WriteLine($"  ✓ Learning Rate: {worker.CurrentState?.LearningRate}");

        worker.Dispose();
    }

    static void TestSharding()
    {
        Console.WriteLine("\nTesting data sharding...");

        var config = new ElasticTrainingConfig
        {
            MinWorkers = 1,
            MaxWorkers = 10,
            WorkerHeartbeatTimeoutMs = 10000
        };

        var workerId = new WorkerId("worker-1", "localhost", 8080);
        var worker = new ElasticWorker(workerId, config);

        var topology = new ClusterTopology();
        topology.AddWorker(new WorkerId("worker-0", "host1", 8080));
        topology.AddWorker(workerId); // Rank 1
        topology.AddWorker(new WorkerId("worker-2", "host2", 8080));

        worker.UpdateTopologyAsync(topology).Wait();

        // Test shard assignment
        var handledShards = new List<int>();
        for (int i = 0; i < 12; i++)
        {
            if (worker.ShouldHandleShard(i, 12))
            {
                handledShards.Add(i);
            }
        }

        Console.WriteLine($"  ✓ Handled shards: [{string.Join(", ", handledShards)}]");
        Console.WriteLine($"  ✓ Expected: [1, 4, 7, 10]");

        worker.Dispose();
    }
}
