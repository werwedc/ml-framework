# Spec: FSDP Mixed Precision Integration

## Overview
Integrate FSDP with Automatic Mixed Precision (AMP) to further reduce memory usage.

## Requirements

### 1. FSDPMixedPrecisionConfig Class
Extend FSDPConfig to support mixed precision:

```csharp
public class FSDPMixedPrecisionConfig
{
    /// <summary>Whether mixed precision is enabled</summary>
    public bool Enabled { get; set; } = true;

    /// <summary>Forward pass data type (FP16 or BF16)</summary>
    public TensorDataType ForwardDType { get; set; } = TensorDataType.Float16;

    /// <summary>Backward pass data type (FP32 for stability)</summary>
    public TensorDataType BackwardDType { get; set; } = TensorDataType.Float32;

    /// <summary>Whether to use loss scaling</summary>
    public bool UseLossScaling { get; set; } = true;

    /// <summary>Initial loss scale</summary>
    public float InitialLossScale { get; set; } = 2.0f;

    /// <summary>Minimum loss scale</summary>
    public float MinLossScale { get; set; } = 1.0f;

    /// <summary>Maximum loss scale</summary>
    public float MaxLossScale { get; set; } = 65536.0f;

    /// <summary>Loss scale growth factor</summary>
    public float LossScaleGrowthFactor { get; set; } = 2.0f;

    /// <summary>Loss scale backoff factor</summary>
    public float LossScaleBackoffFactor { get; set; } = 0.5f;

    /// <summary>Number of steps without overflow before increasing loss scale</summary>
    public int LossScaleSteps { get; set; } = 2000;

    /// <summary>Validate configuration</summary>
    public void Validate()
    {
        if (ForwardDType != TensorDataType.Float16 && ForwardDType != TensorDataType.BFloat16)
        {
            throw new ArgumentException("ForwardDType must be Float16 or BFloat16", nameof(ForwardDType));
        }

        if (BackwardDType != TensorDataType.Float32)
        {
            throw new ArgumentException("BackwardDType must be Float32 for stability", nameof(BackwardDType));
        }

        if (InitialLossScale < MinLossScale || InitialLossScale > MaxLossScale)
        {
            throw new ArgumentException($"InitialLossScale must be between {MinLossScale} and {MaxLossScale}", nameof(InitialLossScale));
        }
    }
}
```

### 2. FSDPMixedPrecisionManager Class
Create a manager for mixed precision operations:

```csharp
public class FSDPMixedPrecisionManager : IDisposable
{
    private readonly FSDPMixedPrecisionConfig _config;
    private readonly FSDP _fsdp;
    private float _currentLossScale;
    private int _stepsSinceOverflow;

    /// <summary>
    /// Initialize mixed precision manager for FSDP.
    /// </summary>
    /// <param name="config">Mixed precision configuration</param>
    /// <param name="fsdp">FSDP wrapper instance</param>
    public FSDPMixedPrecisionManager(FSDPMixedPrecisionConfig config, FSDP fsdp)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _fsdp = fsdp ?? throw new ArgumentNullException(nameof(fsdp));

        _config.Validate();

        _currentLossScale = _config.InitialLossScale;
        _stepsSinceOverflow = 0;
    }

    /// <summary>
    /// Convert gathered parameters to mixed precision for forward pass.
    /// </summary>
    /// <param name="gatheredParam">Full gathered parameter</param>
    /// <returns>Parameter in mixed precision</returns>
    public Tensor ConvertToMixedPrecision(Tensor gatheredParam)
    {
        if (gatheredParam == null)
            throw new ArgumentNullException(nameof(gatheredParam));

        // Convert to forward data type (FP16/BF16)
        return CastTensor(gatheredParam, _config.ForwardDType);
    }

    /// <summary>
    /// Convert mixed precision gradients back to FP32 for stability.
    /// </summary>
    /// <param name="mixedPrecisionGrad">Gradient in mixed precision</param>
    /// <returns>Gradient in FP32</returns>
    public Tensor ConvertGradientToFP32(Tensor mixedPrecisionGrad)
    {
        if (mixedPrecisionGrad == null)
            throw new ArgumentNullException(nameof(mixedPrecisionGrad));

        // Convert to backward data type (FP32)
        return CastTensor(mixedPrecisionGrad, _config.BackwardDType);
    }

    /// <summary>
    /// Cast a tensor to a different data type.
    /// </summary>
    private Tensor CastTensor(Tensor tensor, TensorDataType targetDType)
    {
        if (tensor.DataType == targetDType)
            return tensor; // No conversion needed

        // Create new tensor with target data type
        var result = Tensor.Zeros(tensor.Shape, targetDType);

        // Cast data
        for (int i = 0; i < tensor.Size; i++)
        {
            result.Data[i] = (float)tensor.Data[i];
        }

        return result;
    }

    /// <summary>
    /// Apply loss scaling to the loss tensor.
    /// </summary>
    /// <param name="loss">Loss tensor to scale</param>
    /// <returns>Scaled loss tensor</returns>
    public Tensor ScaleLoss(Tensor loss)
    {
        if (!_config.UseLossScaling)
            return loss;

        var scaledLoss = loss.Clone();
        for (int i = 0; i < scaledLoss.Size; i++)
        {
            scaledLoss.Data[i] *= _currentLossScale;
        }

        return scaledLoss;
    }

    /// <summary>
    /// Check for overflow in gradients and adjust loss scale accordingly.
    /// </summary>
    /// <param name="gradients">Gradients to check</param>
    /// <returns>True if overflow was detected</returns>
    public bool CheckOverflow(Dictionary<string, Tensor> gradients)
    {
        if (!_config.UseLossScaling)
            return false;

        bool overflow = false;

        foreach (var grad in gradients.Values)
        {
            if (grad == null)
                continue;

            // Check for NaN or Inf
            for (int i = 0; i < grad.Size; i++)
            {
                var val = grad.Data[i];
                if (float.IsNaN(val) || float.IsInfinity(val))
                {
                    overflow = true;
                    break;
                }
            }

            if (overflow)
                break;
        }

        if (overflow)
        {
            // Back off loss scale
            _currentLossScale = Math.Max(
                _config.MinLossScale,
                _currentLossScale * _config.LossScaleBackoffFactor
            );
            _stepsSinceOverflow = 0;
        }
        else
        {
            _stepsSinceOverflow++;

            // Increase loss scale if enough steps without overflow
            if (_stepsSinceOverflow >= _config.LossScaleSteps)
            {
                _currentLossScale = Math.Min(
                    _config.MaxLossScale,
                    _currentLossScale * _config.LossScaleGrowthFactor
                );
                _stepsSinceOverflow = 0;
            }
        }

        return overflow;
    }

    /// <summary>
    /// Get the current loss scale.
    /// </summary>
    public float CurrentLossScale => _currentLossScale;

    /// <summary>
    /// Reset the loss scale to the initial value.
    /// </summary>
    public void ResetLossScale()
    {
        _currentLossScale = _config.InitialLossScale;
        _stepsSinceOverflow = 0;
    }

    /// <summary>
    /// Dispose of resources.
    /// </summary>
    public void Dispose()
    {
        // No resources to dispose
    }
}
```

### 3. FSDPAmpIntegration Class
Create integration points for AMP with FSDP:

```csharp
public class FSDPAmpIntegration
{
    private readonly FSDP _fsdp;
    private readonly FSDPMixedPrecisionManager _mpManager;

    /// <summary>
    /// Initialize AMP integration for FSDP.
    /// </summary>
    /// <param name="fsdp">FSDP wrapper instance</param>
    /// <param name="mpConfig">Mixed precision configuration</param>
    public FSDPAmpIntegration(FSDP fsdp, FSDPMixedPrecisionConfig mpConfig = null)
    {
        _fsdp = fsdp ?? throw new ArgumentNullException(nameof(fsdp));

        var config = mpConfig ?? new FSDPMixedPrecisionConfig();
        _mpManager = new FSDPMixedPrecisionManager(config, fsdp);
    }

    /// <summary>
    /// Convert sharded parameters to mixed precision after gathering.
    /// </summary>
    /// <param name="shardingUnit">Sharding unit with gathered parameter</param>
    public void ApplyMixedPrecision(FSDPShardingUnit shardingUnit)
    {
        if (shardingUnit == null)
            throw new ArgumentNullException(nameof(shardingUnit));

        if (shardingUnit.GatheredParameter == null)
            throw new InvalidOperationException("Parameter must be gathered first");

        // Convert to mixed precision
        var mpParam = _mpManager.ConvertToMixedPrecision(shardingUnit.GatheredParameter);

        // Replace gathered parameter with mixed precision version
        shardingUnit.GatheredParameter.Dispose();
        shardingUnit.GatheredParameter = mpParam;
    }

    /// <summary>
    /// Convert gradients back to FP32 before scattering.
    /// </summary>
    /// <param name="shardingUnit">Sharding unit with gradients</param>
    public void ApplyGradientFP32(FSDPShardingUnit shardingUnit)
    {
        if (shardingUnit == null)
            throw new ArgumentNullException(nameof(shardingUnit));

        if (shardingUnit.LocalGradient == null)
            throw new InvalidOperationException("Gradients not computed yet");

        // Convert to FP32
        var fp32Grad = _mpManager.ConvertGradientToFP32(shardingUnit.LocalGradient);

        // Replace gradient with FP32 version
        shardingUnit.LocalGradient.Dispose();
        shardingUnit.LocalGradient = fp32Grad;
    }

    /// <summary>
    /// Apply loss scaling to the loss.
    /// </summary>
    /// <param name="loss">Loss tensor</param>
    /// <returns>Scaled loss tensor</returns>
    public Tensor ScaleLoss(Tensor loss)
    {
        return _mpManager.ScaleLoss(loss);
    }

    /// <summary>
    /// Check for overflow in gradients and adjust loss scale.
    /// </summary>
    /// <param name="gradients">Gradients to check</param>
    /// <returns>True if overflow was detected</returns>
    public bool CheckOverflow(Dictionary<string, Tensor> gradients)
    {
        return _mpManager.CheckOverflow(gradients);
    }

    /// <summary>
    /// Get the mixed precision manager.
    /// </summary>
    public FSDPMixedPrecisionManager Manager => _mpManager;
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/FSDPMixedPrecision.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- `MLFramework.Distributed.FSDP.FSDP`
- `MLFramework.Distributed.FSDP.FSDPShardingUnit`
- `RitterFramework.Core.Tensor`

## Implementation Notes
1. Use FP16/BF16 for forward pass to reduce memory
2. Use FP32 for backward pass to maintain stability
3. Implement dynamic loss scaling to prevent underflow
4. Convert parameters after gathering, before forward pass
5. Convert gradients after backward pass, before scattering

## Testing Requirements
- Test parameter conversion to mixed precision
- Test gradient conversion back to FP32
- Test loss scaling
- Test overflow detection and loss scale adjustment
- Test dynamic loss scaling (growth and backoff)
- Test edge cases (NaN/Inf gradients)
- Test with different data types (FP16 vs BF16)

## Estimated Time
45 minutes
