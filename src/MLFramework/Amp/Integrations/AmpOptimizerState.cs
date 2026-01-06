using MLFramework.Core;

namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// State information for AMP-aware optimizers
    /// </summary>
    public class AmpOptimizerState
    {
        /// <summary>
        /// Gets the underlying optimizer state
        /// </summary>
        public object OptimizerState { get; }

        /// <summary>
        /// Gets the GradScaler state
        /// </summary>
        public object ScalerState { get; }

        /// <summary>
        /// Gets the parameter dtype
        /// </summary>
        public DataType ParameterDtype { get; }

        /// <summary>
        /// Gets the gradient dtype
        /// </summary>
        public DataType GradientDtype { get; }

        /// <summary>
        /// Creates a new AmpOptimizerState
        /// </summary>
        public AmpOptimizerState(
            object optimizerState,
            object scalerState,
            DataType parameterDtype,
            DataType gradientDtype)
        {
            OptimizerState = optimizerState;
            ScalerState = scalerState;
            ParameterDtype = parameterDtype;
            GradientDtype = gradientDtype;
        }

        /// <summary>
        /// Creates a default AmpOptimizerState
        /// </summary>
        public static AmpOptimizerState CreateDefault(DataType parameterDtype)
        {
            return new AmpOptimizerState(
                optimizerState: null,
                scalerState: null,
                parameterDtype: parameterDtype,
                gradientDtype: DataType.Float32
            );
        }
    }
}
