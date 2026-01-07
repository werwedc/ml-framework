using MlFramework.Inference.Batching;
using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Phases;
using MlFramework.Inference.Scheduling;
using MLFramework.Core;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Inference.Batching;

public class ContinuousBatchManagerTests : IDisposable
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly BlockTable _blockTable;
    private readonly PhaseManager _phaseManager;
    private readonly PagedAttentionScheduler _scheduler;
    private readonly Device _device;

    public ContinuousBatchManagerTests()
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

        var prefillOrchestrator = new PrefillOrchestrator(_blockManager, _blockTable);
        var decodeOrchestrator = new DecodeOrchestrator(_blockManager, _blockTable);
        _phaseManager = new PhaseManager(prefillOrchestrator, decodeOrchestrator);
        _scheduler = new PagedAttentionScheduler(_blockManager);
    }

    public void Dispose()
    {
        _blockManager.Dispose();
    }

    [Fact]
    public void TryAddSequence_WithCapacity_ReturnsTrue()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        var result = manager.TryAddSequence(1);

        Assert.True(result);
    }

    [Fact]
    public void TryAddSequence_WhenFull_ReturnsFalse()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 2);

        manager.TryAddSequence(1);
        manager.TryAddSequence(2);
        var result = manager.TryAddSequence(3);

        Assert.False(result);
    }

    [Fact]
    public void TryAddSequence_WithSameId_ReturnsTrue()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        manager.TryAddSequence(1);
        var result = manager.TryAddSequence(1);

        Assert.True(result);
    }

    [Fact]
    public void RemoveSequence_RemovesFromBatch()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        manager.TryAddSequence(1);
        manager.RemoveSequence(1);
        var batch = manager.GetCurrentBatch();

        Assert.DoesNotContain(1, batch);
    }

    [Fact]
    public void GetCurrentBatch_ReturnsDefensiveCopy()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        manager.TryAddSequence(1);
        var batch1 = manager.GetCurrentBatch();
        var batch2 = manager.GetCurrentBatch();

        // Should be different instances
        Assert.NotSame(batch1, batch2);
    }

    [Fact]
    public void GetPrefillSequences_ReturnsOnlyPrefill()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        manager.TryAddSequence(1);
        manager.TryAddSequence(2);
        _phaseManager.SetPhase(1, SequencePhase.Prefill);
        _phaseManager.SetPhase(2, SequencePhase.Decode);

        var prefillSeqs = manager.GetPrefillSequences();

        Assert.Single(prefillSeqs);
        Assert.Contains(1, prefillSeqs);
        Assert.DoesNotContain(2, prefillSeqs);
    }

    [Fact]
    public void GetDecodeSequences_ReturnsOnlyDecode()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        manager.TryAddSequence(1);
        manager.TryAddSequence(2);
        _phaseManager.SetPhase(1, SequencePhase.Prefill);
        _phaseManager.SetPhase(2, SequencePhase.Decode);

        var decodeSeqs = manager.GetDecodeSequences();

        Assert.Single(decodeSeqs);
        Assert.Contains(2, decodeSeqs);
        Assert.DoesNotContain(1, decodeSeqs);
    }

    [Fact]
    public void TransitionToDecode_UpdatesPhase()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        manager.TryAddSequence(1);
        _phaseManager.SetPhase(1, SequencePhase.Prefill);
        manager.TransitionToDecode(1);

        Assert.True(_phaseManager.IsDecoding(1));
    }

    [Fact]
    public void CanAcceptMoreSequences_WhenFull_ReturnsFalse()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 2);

        manager.TryAddSequence(1);
        manager.TryAddSequence(2);
        var result = manager.CanAcceptMoreSequences();

        Assert.False(result);
    }

    [Fact]
    public void CanAcceptMoreSequences_WithSpace_ReturnsTrue()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        manager.TryAddSequence(1);
        var result = manager.CanAcceptMoreSequences();

        Assert.True(result);
    }

    [Fact]
    public void GetStats_ReturnsCorrectStats()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        manager.TryAddSequence(1);
        manager.TryAddSequence(2);
        _phaseManager.SetPhase(1, SequencePhase.Prefill);
        _phaseManager.SetPhase(2, SequencePhase.Decode);

        var stats = manager.GetStats();

        Assert.Equal(2, stats.TotalSequences);
        Assert.Equal(1, stats.PrefillSequences);
        Assert.Equal(1, stats.DecodeSequences);
        Assert.Equal(10, stats.MaxBatchSize);
        Assert.Equal(0.2, stats.Utilization);
    }

    [Fact]
    public void Clear_RemovesAllSequences()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        manager.TryAddSequence(1);
        manager.TryAddSequence(2);
        manager.Clear();
        var batch = manager.GetCurrentBatch();

        Assert.Empty(batch);
    }

    [Fact]
    public void ToString_ProvidesNiceFormat()
    {
        var manager = new ContinuousBatchManager(
            _blockManager, _blockTable, _phaseManager, _scheduler, maxBatchSize: 10);

        manager.TryAddSequence(1);
        manager.TryAddSequence(2);
        _phaseManager.SetPhase(1, SequencePhase.Prefill);
        _phaseManager.SetPhase(2, SequencePhase.Decode);

        var stats = manager.GetStats();
        var str = stats.ToString();

        Assert.Contains("BatchStats", str);
        Assert.Contains("Total=2/10", str);
    }
}
