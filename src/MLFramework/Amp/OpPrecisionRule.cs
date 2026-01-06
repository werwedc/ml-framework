using MLFramework.Core;
using System;

namespace MLFramework.Amp
{
    /// <summary>
    /// Defines precision rules for a specific operation
    /// </summary>
    public class OpPrecisionRule
    {
        /// <summary>
        /// Gets the operation type this rule applies to
        /// </summary>
        public Type OperationType { get; }

        /// <summary>
        /// Gets the forward pass precision policy
        /// </summary>
        public OpPrecision ForwardPrecision { get; }

        /// <summary>
        /// Gets the backward pass precision policy
        /// </summary>
        public OpPrecision BackwardPrecision { get; }

        /// <summary>
        /// Gets or sets custom forward dtype (if ForwardPrecision is Custom)
        /// </summary>
        public DataType? CustomForwardDtype { get; set; }

        /// <summary>
        /// Gets or sets custom backward dtype (if BackwardPrecision is Custom)
        /// </summary>
        public DataType? CustomBackwardDtype { get; set; }

        /// <summary>
        /// Gets or sets the priority (higher = more important)
        /// </summary>
        public int Priority { get; set; }

        /// <summary>
        /// Creates a new OpPrecisionRule
        /// </summary>
        public OpPrecisionRule(
            Type operationType,
            OpPrecision forwardPrecision,
            OpPrecision backwardPrecision = OpPrecision.Keep,
            int priority = 0)
        {
            OperationType = operationType ?? throw new ArgumentNullException(nameof(operationType));
            ForwardPrecision = forwardPrecision;
            BackwardPrecision = backwardPrecision;
            Priority = priority;
        }

        /// <summary>
        /// Gets the actual forward dtype based on AMP config
        /// </summary>
        public DataType GetForwardDtype(AmpConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            return ForwardPrecision switch
            {
                OpPrecision.Higher => config.HigherPrecision,
                OpPrecision.Lower => config.TargetPrecision,
                OpPrecision.Custom => CustomForwardDtype ?? config.TargetPrecision,
                OpPrecision.Keep => config.TargetPrecision, // Keep is context-dependent, default to target
                _ => config.TargetPrecision
            };
        }

        /// <summary>
        /// Gets the actual backward dtype based on AMP config
        /// </summary>
        public DataType GetBackwardDtype(AmpConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            return BackwardPrecision switch
            {
                OpPrecision.Higher => config.HigherPrecision,
                OpPrecision.Lower => config.TargetPrecision,
                OpPrecision.Custom => CustomBackwardDtype ?? config.TargetPrecision,
                OpPrecision.Keep => config.TargetPrecision, // Keep is context-dependent, default to target
                _ => config.TargetPrecision
            };
        }
    }
}
