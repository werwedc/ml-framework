namespace MachineLearning.Distributed.Data;

using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Enums;
using MachineLearning.Distributed.Models;

/// <summary>
/// Handles data redistribution when topology changes
/// </summary>
public class DataRedistributor
{
    private readonly ElasticTrainingConfig _config;

    public DataRedistributor(ElasticTrainingConfig config)
    {
        _config = config;
        _config.Validate();
    }

    /// <summary>
    /// Calculate redistribution plan based on old and new cluster state
    /// </summary>
    public DataRedistributionPlan CalculatePlan(
        ClusterTopology oldState,
        ClusterTopology newState,
        int totalDatasetSize)
    {
        var plan = new DataRedistributionPlan();

        if (_config.RedistributionType == RedistributionType.FullReshuffle)
        {
            CalculateFullReshufflePlan(oldState, newState, totalDatasetSize, plan);
        }
        else
        {
            CalculateIncrementalPlan(oldState, newState, totalDatasetSize, plan);
        }

        return plan;
    }

    /// <summary>
    /// Transfer data shards between workers
    /// </summary>
    public async Task TransferDataShardsAsync(
        WorkerId source,
        WorkerId destination,
        DataShard shard)
    {
        // In a full implementation, this would:
        // 1. Serialize the data shard from source worker
        // 2. Transfer over network to destination worker
        // 3. Deserialize and integrate into destination's local data

        await Task.CompletedTask;
    }

    /// <summary>
    /// Validate redistribution completeness across the cluster
    /// </summary>
    public bool ValidateRedistribution(ClusterTopology state, DataRedistributionPlan plan)
    {
        // Check that all workers have been assigned the correct number of shards
        var expectedShardsPerWorker = plan.TotalShards / state.WorldSize;
        var remainder = plan.TotalShards % state.WorldSize;

        foreach (var worker in state.Workers)
        {
            if (!plan.WorkerAssignments.TryGetValue(worker, out var shards))
            {
                return false;
            }

            var workerRank = state.Workers.IndexOf(worker);
            var expectedCount = workerRank < remainder ? expectedShardsPerWorker + 1 : expectedShardsPerWorker;

            if (shards.Count != expectedCount)
            {
                return false;
            }
        }

        return true;
    }

    private void CalculateFullReshufflePlan(
        ClusterTopology oldState,
        ClusterTopology newState,
        int totalDatasetSize,
        DataRedistributionPlan plan)
    {
        var oldWorkers = oldState.Workers;
        var newWorkers = newState.Workers;
        var oldWorkerCount = oldState.WorldSize;
        var newWorkerCount = newState.WorldSize;

        // Calculate shard assignments for old topology
        var oldAssignments = new Dictionary<WorkerId, List<DataShard>>();
        for (int i = 0; i < totalDatasetSize; i++)
        {
            var oldWorker = oldWorkers[i % oldWorkerCount];
            if (!oldAssignments.ContainsKey(oldWorker))
            {
                oldAssignments[oldWorker] = new List<DataShard>();
            }

            var shardSize = CalculateShardSize(totalDatasetSize, oldWorkerCount, i);
            oldAssignments[oldWorker].Add(new DataShard(i, i, i + shardSize));
        }

        // Calculate shard assignments for new topology
        for (int i = 0; i < totalDatasetSize; i++)
        {
            var shardId = i;
            var shardSize = CalculateShardSize(totalDatasetSize, newWorkerCount, i);
            var shard = new DataShard(shardId, i, i + shardSize);
            var newWorker = newWorkers[i % newWorkerCount];

            if (!plan.WorkerAssignments.ContainsKey(newWorker))
            {
                plan.WorkerAssignments[newWorker] = new List<DataShard>();
            }
            plan.WorkerAssignments[newWorker].Add(shard);

            // Find which worker currently has this data and add transfer
            var oldWorker = oldWorkers[i % oldWorkerCount];
            if (!newWorkers.Contains(oldWorker) || newWorker != oldWorker)
            {
                plan.Transfers.Add(new DataTransfer
                {
                    SourceWorker = oldWorker,
                    DestinationWorker = newWorker,
                    Shard = shard,
                    Priority = CalculateTransferPriority(shardId, totalDatasetSize),
                    EstimatedCompletionTime = DateTime.UtcNow.AddSeconds(10)
                });
            }
        }

        plan.TotalShards = totalDatasetSize;
    }

    private void CalculateIncrementalPlan(
        ClusterTopology oldState,
        ClusterTopology newState,
        int totalDatasetSize,
        DataRedistributionPlan plan)
    {
        var oldWorkers = oldState.Workers;
        var newWorkers = newState.Workers;
        var newWorkerCount = newState.WorldSize;

        // Keep existing assignments for workers that are still present
        var existingWorkers = oldWorkers.Intersect(newWorkers).ToList();

        // Calculate which shards need to be redistributed
        var redistributedShards = new List<(int shardIndex, WorkerId newOwner)>();

        for (int i = 0; i < totalDatasetSize; i++)
        {
            var newWorker = newWorkers[i % newWorkerCount];
            var oldWorker = oldWorkers[i % oldState.WorldSize];

            // Shard needs redistribution if:
            // 1. It was owned by a worker that left
            // 2. It's owned by a worker that still exists but assignment changed
            if (!newWorkers.Contains(oldWorker) || oldWorker != newWorker)
            {
                redistributedShards.Add((i, newWorker));

                if (!plan.WorkerAssignments.ContainsKey(newWorker))
                {
                    plan.WorkerAssignments[newWorker] = new List<DataShard>();
                }

                var shardSize = CalculateShardSize(totalDatasetSize, newWorkerCount, i);
                var shard = new DataShard(i, i, i + shardSize);
                plan.WorkerAssignments[newWorker].Add(shard);

                if (newWorkers.Contains(oldWorker))
                {
                    plan.Transfers.Add(new DataTransfer
                    {
                        SourceWorker = oldWorker,
                        DestinationWorker = newWorker,
                        Shard = shard,
                        Priority = CalculateTransferPriority(i, totalDatasetSize),
                        EstimatedCompletionTime = DateTime.UtcNow.AddSeconds(5)
                    });
                }
            }
        }

        plan.TotalShards = totalDatasetSize;
    }

    private int CalculateShardSize(int totalSize, int workerCount, int index)
    {
        var baseSize = totalSize / workerCount;
        var remainder = totalSize % workerCount;
        return index < remainder ? baseSize + 1 : baseSize;
    }

    private int CalculateTransferPriority(int shardIndex, int totalShards)
    {
        // Higher priority for shards that appear earlier in the training sequence
        // Invert the index so lower indices get higher priority
        return totalShards - shardIndex;
    }
}
