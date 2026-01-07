using MlFramework.Inference.Batching;
using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Kernels;
using MlFramework.Inference.PagedAttention.Phases;
using MlFramework.Inference.Scheduling;
using MLFramework.Core;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Inference.Batching;

public class IterationSchedulerTests : IDisposable
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly BlockTable _blockTable;
    private readonly PhaseManager _phaseManager;
    private readonly PagedAttentionScheduler _pagedScheduler;
    private readonly MemoryAwareScheduler _memoryScheduler;
    private readonly PrefillOrchestrator _prefillOrchestrator;
    private readonly DecodeOrchestrator _decodeOrchestrator;
    private readonly Device _device;

    public IterationSchedulerTests()
    {
        _device = Device.CPU();
        _blockManager = new KVCacheBlockManager(
            totalBlocks: 100,
            blockSize: 16,
            headDim: 128,
            numLayers: 32,
            numAttentionHeads: 32,
            device: _device
        );
        _blockTable = new BlockTable(_blockManager);
        _prefillOrchestrator = new PrefillOrchestrator(_blockManager, _blockTable);
        _decodeOrchestrator = new DecodeOrchestrator(_blockManager, _blockTable);
        _phaseManager = new PhaseManager(_prefillOrchestrator, _decodeOrchestrator);
        _pagedScheduler = new PagedAttentionScheduler(_blockManager);
        _memoryScheduler = new MemoryAwareScheduler(_pagedScheduler);
    }

    public void Dispose()
    {
        _blockManager.Dispose();
    }

    [Fact]
    public void RunIteration_WithNoSequences_DoesNothing()
    {
        var batchManager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _pagedScheduler, maxBatchSize: 10);

        var scheduler = new IterationScheduler(
            batchManager,
            _memoryScheduler,
            _prefillOrchestrator,
            _decodeOrchestrator
        );

        var queries = new Dictionary<int, Tensor>();
        scheduler.RunIteration(queries);

        var stats = batchManager.GetStats();
        Assert.Equal(0, stats.TotalSequences);
    }

    [Fact]
    public void RunIteration_WithSequences_ProcessesBatch()
    {
        var batchManager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _pagedScheduler, maxBatchSize: 10);

        var scheduler = new IterationScheduler(
            batchManager,
            _memoryScheduler,
            _prefillOrchestrator,
            _decodeOrchestrator
        );

        // Add some sequences
        batchManager.TryAddSequence(1);
        batchManager.TryAddSequence(2);
        _phaseManager.SetPhase(1, SequencePhase.Decode);
        _phaseManager.SetPhase(2, SequencePhase.Decode);

        var queries = new Dictionary<int, Tensor>
        {
            { 1, CreateRandomQuery() },
            { 2, CreateRandomQuery() }
        };

        // This should process the sequences
        scheduler.RunIteration(queries);
    }

    [Fact]
    public void RunIteration_CallsIterationCallback()
    {
        var batchManager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _pagedScheduler, maxBatchSize: 10);

        BatchStats? capturedStats = null;
        var callback = new Action<BatchStats>(stats => { capturedStats = stats; });

        var scheduler = new IterationScheduler(
            batchManager,
            _memoryScheduler,
            _prefillOrchestrator,
            _decodeOrchestrator,
            callback
        );

        batchManager.TryAddSequence(1);
        _phaseManager.SetPhase(1, SequencePhase.Decode);
        var queries = new Dictionary<int, Tensor> { { 1, CreateRandomQuery() } };

        scheduler.RunIteration(queries);

        Assert.NotNull(capturedStats);
        Assert.Equal(1, capturedStats.Value.TotalSequences);
    }

    [Fact]
    public void RunIteration_AdmitsNewRequests()
    {
        var batchManager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _pagedScheduler, maxBatchSize: 10);

        var scheduler = new IterationScheduler(
            batchManager,
            _memoryScheduler,
            _prefillOrchestrator,
            _decodeOrchestrator
        );

        // Add a request to the memory scheduler queue
        var request = new PendingRequest
        {
            RequestId = 1,
            EstimatedMaxTokens = 100,
            QueuedAt = DateTime.UtcNow,
            Priority = 0
        };
        _memoryScheduler.EnqueueRequest(request);

        var queries = new Dictionary<int, Tensor>();

        // Run iteration - should admit the request
        scheduler.RunIteration(queries);

        var batch = batchManager.GetCurrentBatch();
        Assert.Contains(1, batch);
    }

    [Fact]
    public void RunIteration_ProcessesPrefillAndDecodeSeparately()
    {
        var batchManager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _pagedScheduler, maxBatchSize: 10);

        var scheduler = new IterationScheduler(
            batchManager,
            _memoryScheduler,
            _prefillOrchestrator,
            _decodeOrchestrator
        );

        // Add sequences in both phases
        batchManager.TryAddSequence(1);
        batchManager.TryAddSequence(2);
        _phaseManager.SetPhase(1, SequencePhase.Prefill);
        _phaseManager.SetPhase(2, SequencePhase.Decode);

        var queries = new Dictionary<int, Tensor>
        {
            { 1, CreateRandomQuery() },
            { 2, CreateRandomQuery() }
        };

        scheduler.RunIteration(queries);

        // After prefill, sequence 1 should be in decode phase
        Assert.True(_phaseManager.IsDecoding(1));
    }

    [Fact]
    public void IterationSchedulerConfig_HasDefaultValues()
    {
        var config = new IterationSchedulerConfig();

        Assert.Equal(32, config.MaxBatchSize);
        Assert.Equal(1, config.MinDecodeSequences);
        Assert.True(config.PrioritizePrefill);
        Assert.Equal(4, config.MaxPrefillPerIteration);
    }

    [Fact]
    public void IterationSchedulerConfig_CanBeModified()
    {
        var config = new IterationSchedulerConfig
        {
            MaxBatchSize = 64,
            MinDecodeSequences = 2,
            PrioritizePrefill = false,
            MaxPrefillPerIteration = 8
        };

        Assert.Equal(64, config.MaxBatchSize);
        Assert.Equal(2, config.MinDecodeSequences);
        Assert.False(config.PrioritizePrefill);
        Assert.Equal(8, config.MaxPrefillPerIteration);
    }

    private Tensor CreateRandomQuery()
    {
        var random = new Random(42);
        var data = new float[1 * 1 * 32 * 128];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)random.NextDouble();
        }
        return new Tensor(data, new[] { 1, 1, 32, 128 }, _device);
    }
}
