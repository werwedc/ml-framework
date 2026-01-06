using RitterFramework.Core.Tensor;
using System.Collections.Generic;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Helper class to hold gathered states from all ranks during checkpoint operations.
    /// </summary>
    public class GatheredStates
    {
        /// <summary>Gathered parameters from all ranks</summary>
        public Dictionary<string, Tensor> Parameters { get; set; }

        /// <summary>Gathered optimizer states from all ranks</summary>
        public Dictionary<string, OptimizerState> OptimizerStates { get; set; }

        /// <summary>
        /// Create a new GatheredStates instance.
        /// </summary>
        public GatheredStates()
        {
            Parameters = new Dictionary<string, Tensor>();
            OptimizerStates = new Dictionary<string, OptimizerState>();
        }
    }
}
