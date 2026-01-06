using MLFramework.Core;
using MLFramework.Amp;

namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Context for AMP-aware autograd operations
    /// </summary>
    public class AmpAutogradContext
    {
        /// <summary>
        /// Gets or sets the current AutoCast mode
        /// </summary>
        public AutoCastMode Mode { get; set; }

        /// <summary>
        /// Gets or sets the operation precision registry
        /// </summary>
        public AmpRegistry? Registry { get; set; }

        /// <summary>
        /// Gets or sets the loss scaler
        /// </summary>
        public ILossScaler? LossScaler { get; set; }

        /// <summary>
        /// Gets whether gradient unscaling is needed
        /// </summary>
        public bool NeedsGradientUnscaling { get; set; }

        /// <summary>
        /// Creates a new AmpAutogradContext
        /// </summary>
        public AmpAutogradContext(
            AutoCastMode mode = AutoCastMode.Bf16,
            AmpRegistry? registry = null,
            ILossScaler? lossScaler = null,
            bool needsGradientUnscaling = false)
        {
            Mode = mode;
            Registry = registry;
            LossScaler = lossScaler;
            NeedsGradientUnscaling = needsGradientUnscaling;
        }
    }
}
