# Spec: LoRAEmbedding Layer Implementation

## Overview
Implement the LoRAEmbedding layer, which wraps a standard Embedding layer and injects low-rank adapter matrices for token embeddings. This enables parameter-efficient fine-tuning of vocabulary embeddings in language models.

## Implementation Details

### 1. LoRAEmbedding Class
**File**: `src/LoRA/LoRAEmbedding.cs`

```csharp
public class LoRAEmbedding : LoRAAdapterBase, IModule
{
    private readonly Embedding _embeddingLayer;
    private readonly ITensor _loraA; // Rank x EmbeddingDim
    private readonly ITensor _loraB; // NumTokens x Rank
    private readonly float _dropoutRate;
    private readonly Random? _dropoutRandom;

    public int NumTokens => _embeddingLayer.NumEmbeddings;
    public int EmbeddingDim => _embeddingLayer.EmbeddingDim;

    public LoRAEmbedding(Embedding embeddingLayer, int rank, float alpha,
                         LoRAInitializationStrategy initialization = LoRAInitializationStrategy.Standard,
                         float dropout = 0.0f)
        : base(embeddingLayer, rank, alpha)
    {
        _embeddingLayer = embeddingLayer ?? throw new ArgumentNullException(nameof(embeddingLayer));
        _dropoutRate = dropout;

        InitializeLoRAMatrices(initialization);

        if (_dropoutRate > 0.0f)
        {
            _dropoutRandom = new Random(42);
        }
    }

    private void InitializeLoRAMatrices(LoRAInitializationStrategy strategy)
    {
        int embeddingDim = _embeddingLayer.EmbeddingDim;
        int numTokens = _embeddingLayer.NumEmbeddings;

        switch (strategy)
        {
            case LoRAInitializationStrategy.Standard:
                _loraA = Tensor.KaimingNormal(new[] { Rank, embeddingDim });
                _loraB = Tensor.Zeros(new[] { numTokens, Rank });
                break;

            case LoRAInitializationStrategy.Xavier:
                _loraA = Tensor.XavierUniform(new[] { Rank, embeddingDim });
                _loraB = Tensor.XavierUniform(new[] { numTokens, Rank });
                break;

            case LoRAInitializationStrategy.Zero:
                _loraA = Tensor.Zeros(new[] { Rank, embeddingDim });
                _loraB = Tensor.Zeros(new[] { numTokens, Rank });
                break;

            default:
                throw new ArgumentException($"Unknown initialization strategy: {strategy}");
        }
    }

    public ITensor Forward(ITensor input)
    {
        // input: [batch, sequence_length] of token IDs

        // Standard embedding lookup
        var output = _embeddingLayer.Forward(input);

        if (!_isEnabled)
            return output;

        // LoRA forward pass for embeddings
        // For each token in input, add low-rank adaptation

        // First, get the base embeddings for all tokens: [num_tokens, emb_dim]
        var baseEmbeddings = _embeddingLayer.Weight;

        // LoRA: E_new = E + (alpha/r) * B * A
        // B: [num_tokens, rank], A: [rank, emb_dim]
        // Result: [num_tokens, emb_dim]

        var deltaEmbeddings = Tensor.MatMul(_loraB, _loraA); // [num_tokens, rank] x [rank, emb_dim]
        deltaEmbeddings = deltaEmbeddings.Mul(ScalingFactor);

        // Create adapted embedding matrix
        var adaptedEmbeddings = baseEmbeddings.Add(deltaEmbeddings);

        // Look up the adapted embeddings for input tokens
        var batchSize = input.Shape[0];
        int seqLength = input.Shape[1];
        var embDim = output.Shape[2];

        // Create output tensor for adapted embeddings
        var adaptedOutput = Tensor.Zeros(new[] { batchSize, seqLength, embDim });

        // For each position, select the appropriate adapted embedding
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < seqLength; i++)
            {
                int tokenId = (int)input[b, i];
                var embSlice = adaptedEmbeddings[tokenId]; // [emb_dim]

                // Apply dropout if in training mode
                if (_dropoutRate > 0.0f && IsTrainingMode)
                {
                    embSlice = ApplyDropoutToEmbedding(embSlice);
                }

                for (int d = 0; d < embDim; d++)
                {
                    adaptedOutput[b, i, d] = embSlice[d];
                }
            }
        }

        // Add LoRA adaptation to base output
        return output.Add(adaptedOutput);
    }

    private ITensor ApplyDropoutToEmbedding(ITensor embedding)
    {
        var mask = Tensor.Random(embedding.Shape, _dropoutRandom);
        mask = Tensor.Where(mask.GreaterThan(_dropoutRate), 1.0f / (1.0f - _dropoutRate), 0.0f);
        return embedding.Mul(mask);
    }

    public bool IsTrainingMode { get; set; } = false;

    public override void FreezeBaseLayer()
    {
        _embeddingLayer.Weight.RequiresGrad = false;
        _isBaseLayerFrozen = true;
    }

    public override void UnfreezeBaseLayer()
    {
        _embeddingLayer.Weight.RequiresGrad = true;
        _isBaseLayerFrozen = false;
    }

    public override IEnumerable<ITensor> TrainableParameters
    {
        get
        {
            if (!_isBaseLayerFrozen)
            {
                yield return _embeddingLayer.Weight;
            }
            yield return _loraA;
            yield return _loraB;
        }
    }

    public override IEnumerable<ITensor> FrozenParameters
    {
        get
        {
            if (_isBaseLayerFrozen)
            {
                yield return _embeddingLayer.Weight;
            }
        }
    }

    public override void MergeAdapter()
    {
        // Backup original weights
        _baseLayerWeightsBackup = _embeddingLayer.Weight.Clone();

        // E_new = E + (alpha/r) * B * A
        var deltaE = Tensor.MatMul(_loraB, _loraA);
        deltaE = deltaE.Mul(ScalingFactor);

        _embeddingLayer.Weight = _embeddingLayer.Weight.Add(deltaE);
    }

    public override void ResetBaseLayer()
    {
        if (_baseLayerWeightsBackup == null)
            throw new InvalidOperationException("No backup available. Cannot reset.");

        _embeddingLayer.Weight = _baseLayerWeightsBackup;
        _baseLayerWeightsBackup = null;
    }

    public override (ITensor? MatrixA, ITensor? MatrixB) GetAdapterWeights()
    {
        return (_loraA, _loraB);
    }

    public override void SetAdapterWeights(ITensor? matrixA, ITensor? matrixB)
    {
        if (matrixA == null || matrixB == null)
            throw new ArgumentNullException("Adapter weights cannot be null");

        int embeddingDim = _embeddingLayer.EmbeddingDim;
        int numTokens = _embeddingLayer.NumEmbeddings;

        // Validate shapes
        if (matrixA.Shape.Length != 2 || matrixA.Shape[0] != Rank || matrixA.Shape[1] != embeddingDim)
            throw new ArgumentException($"Matrix A shape must be [{Rank}, {embeddingDim}]");

        if (matrixB.Shape.Length != 2 || matrixB.Shape[0] != numTokens || matrixB.Shape[1] != Rank)
            throw new ArgumentException($"Matrix B shape must be [{numTokens}, {Rank}]");

        _loraA.CopyFrom(matrixA);
        _loraB.CopyFrom(matrixB);
    }
}
```

### 2. Extension Method
**File**: `src/LoRA/LoRAExtensions.cs`

```csharp
public static class LoRAExtensions
{
    public static LoRAEmbedding AsLoRA(this Embedding embedding, int rank, float alpha,
                                      LoRAInitializationStrategy initialization = LoRAInitializationStrategy.Standard,
                                      float dropout = 0.0f)
    {
        return new LoRAEmbedding(embedding, rank, alpha, initialization, dropout);
    }
}
```

## Testing Requirements

**File**: `tests/LoRA/LoRAEmbeddingTests.cs`

1. **Constructor Tests**
   - Test wrapping various embedding sizes
   - Test different initialization strategies
   - Test vocabulary size handling

2. **Forward Pass Tests**
   - Test output shape matches base layer
   - Test that disabled adapter returns base output
   - Test dropout in training mode
   - Test token lookup with LoRA adaptation

3. **Freeze/Unfreeze Tests**
   - Test correct parameter freezing
   - Test TrainableParameters property

4. **Merge/Reset Tests**
   - Test MergeAdapter correctly updates embedding weights
   - Test ResetBaseLayer restores original weights
   - Verify embedding shape preservation

5. **Adapter Weight Tests**
   - Test GetAdapterWeights returns correct shapes
   - Test SetAdapterWeights validates input shapes

## Dependencies
- `Embedding` layer (existing)
- `Tensor` class (existing)
- `ILoRAAdapter` interface (from spec 001)
- `LoRAAdapterBase` (from spec 001)

## Success Criteria
- LoRAEmbedding correctly wraps Embedding layers
- Forward pass produces expected outputs with LoRA adaptation
- Token lookup works correctly with LoRA
- Adapter weights correctly adapt vocabulary
- All unit tests pass

## Estimated Time
45 minutes

## Notes
- Embedding LoRA is useful for fine-tuning task-specific vocabulary terms
- Consider supporting sparse updates (only update embeddings for tokens in batch)
- Dropout should only apply during training mode
- Merge is destructive - ensure backup capability works
- For very large vocabularies, consider memory-efficient implementations
