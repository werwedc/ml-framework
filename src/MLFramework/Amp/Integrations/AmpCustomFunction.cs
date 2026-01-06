using System;
using MLFramework.Amp;
using MLFramework.Core;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Base class for custom AMP-aware autograd functions
    /// </summary>
    /// <typeparam name="TContext">The context type for the function</typeparam>
    public abstract class AmpCustomFunction<TContext> : AmpAutogradFunction
        where TContext : AmpAutogradContext, new()
    {
        /// <summary>
        /// Creates a new AmpCustomFunction
        /// </summary>
        protected AmpCustomFunction(AmpRegistry? registry = null)
            : base(registry) { }

        /// <summary>
        /// Creates the context for the forward/backward pass
        /// </summary>
        /// <param name="inputs">Input tensors</param>
        /// <returns>The context</returns>
        protected abstract TContext CreateContext(Tensor[] inputs);

        /// <summary>
        /// Forward pass implementation
        /// </summary>
        /// <param name="ctx">The context</param>
        /// <param name="inputs">Input tensors</param>
        /// <returns>Output tensors</returns>
        protected abstract Tensor[] ForwardImpl(TContext ctx, Tensor[] inputs);

        /// <summary>
        /// Backward pass implementation
        /// </summary>
        /// <param name="ctx">The context</param>
        /// <param name="gradOutputs">Gradient outputs</param>
        /// <returns>Gradient inputs</returns>
        protected abstract Tensor[] BackwardImpl(TContext ctx, Tensor[] gradOutputs);

        /// <summary>
        /// Forward pass with manual precision specification
        /// </summary>
        public override Tensor[] ForwardManual(
            Tensor[] inputs,
            DataType forwardDtype,
            DataType backwardDtype)
        {
            var ctx = CreateContext(inputs);
            ctx.Mode = MapDataTypeToMode(forwardDtype);
            ctx.Registry = Registry;
            return ForwardImpl(ctx, inputs);
        }

        /// <summary>
        /// Backward pass with manual precision specification
        /// </summary>
        public override Tensor[] BackwardManual(
            Tensor[] gradOutputs,
            DataType backwardDtype)
        {
            var ctx = new TContext();
            ctx.Mode = MapDataTypeToMode(backwardDtype);
            ctx.Registry = Registry;
            return BackwardImpl(ctx, gradOutputs);
        }

        private AutoCastMode MapDataTypeToMode(DataType dtype)
        {
            return dtype switch
            {
                DataType.Float16 => AutoCastMode.Fp16,
                DataType.BFloat16 => AutoCastMode.Bf16,
                _ => AutoCastMode.None
            };
        }
    }
}
