using System;
using System.Collections.Generic;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Configuration for mixed precision training with FSDP.
    /// </summary>
    public class FSDPMixedPrecisionConfig
    {
        /// <summary>Whether mixed precision is enabled</summary>
        public bool Enabled { get; set; } = true;

        /// <summary>Forward pass data type (FP16 or BF16)</summary>
        public DataType ForwardDType { get; set; } = DataType.Float16;

        /// <summary>Backward pass data type (FP32 for stability)</summary>
        public DataType BackwardDType { get; set; } = DataType.Float32;

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
            if (ForwardDType != DataType.Float16 && ForwardDType != DataType.BFloat16)
            {
                throw new ArgumentException("ForwardDType must be Float16 or BFloat16", nameof(ForwardDType));
            }

            if (BackwardDType != DataType.Float32)
            {
                throw new ArgumentException("BackwardDType must be Float32 for stability", nameof(BackwardDType));
            }

            if (InitialLossScale < MinLossScale || InitialLossScale > MaxLossScale)
            {
                throw new ArgumentException($"InitialLossScale must be between {MinLossScale} and {MaxLossScale}", nameof(InitialLossScale));
            }
        }
    }

    /// <summary>
    /// Manager for mixed precision operations in FSDP.
    /// Handles conversion between data types and dynamic loss scaling.
    /// </summary>
    public class FSDPMixedPrecisionManager : IDisposable
    {
        private readonly FSDPMixedPrecisionConfig _config;
        private readonly FSDP _fsdp;
        private float _currentLossScale;
        private int _stepsSinceOverflow;
        private bool _disposed;

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
        private Tensor CastTensor(Tensor tensor, DataType targetDType)
        {
            if (tensor.Dtype == targetDType)
                return tensor; // No conversion needed

            // Create new tensor with target data type
            var result = Tensor.Zeros(tensor.Shape, targetDType);

            // Cast data
            for (int i = 0; i < tensor.Size; i++)
            {
                result.Data[i] = tensor.Data[i];
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
            if (!_disposed)
            {
                // No resources to dispose
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Integration points for Automatic Mixed Precision (AMP) with FSDP.
    /// </summary>
    public class FSDPAmpIntegration : IDisposable
    {
        private readonly FSDP _fsdp;
        private readonly FSDPMixedPrecisionManager _mpManager;
        private bool _disposed;

        /// <summary>
        /// Initialize AMP integration for FSDP.
        /// </summary>
        /// <param name="fsdp">FSDP wrapper instance</param>
        /// <param name="mpConfig">Mixed precision configuration</param>
        public FSDPAmpIntegration(FSDP fsdp, FSDPMixedPrecisionConfig? mpConfig = null)
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

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _mpManager?.Dispose();
                _disposed = true;
            }
        }
    }
}
