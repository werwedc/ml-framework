using MLFramework.Shapes;
using MLFramework.Shapes.Inference;

namespace MLFramework.Shapes.Inference.Rules
{
    /// <summary>
    /// Shape inference rule for 2D convolution operations.
    /// </summary>
    public class Conv2DRule : ShapeInferenceRuleBase
    {
        /// <summary>
        /// Gets the supported operations.
        /// </summary>
        protected override string[] SupportedOperations => new[] { "Conv2D", "Convolution", "Conv" };

        /// <summary>
        /// Gets the expected input count.
        /// </summary>
        protected override int GetExpectedInputCount(string opName)
        {
            return 2;
        }

        /// <summary>
        /// Infers the output shape for 2D convolution.
        /// Expected input shape: [N, C_in, H_in, W_in]
        /// Expected weight shape: [C_out, C_in, K_h, K_w]
        /// Output shape: [N, C_out, H_out, W_out]
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>The inferred output shape.</returns>
        protected override List<SymbolicShape> InferOutputShapes(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            var inputShape = inputs[0]; // [N, C_in, H_in, W_in]
            var weightShape = inputs[1]; // [C_out, C_in, K_h, K_w]

            // Validate input rank
            if (inputShape.Rank != 4)
            {
                throw new ArgumentException(
                    $"Input for '{opName}' must have rank 4 [N, C, H, W], got rank {inputShape.Rank}");
            }

            // Validate weight rank
            if (weightShape.Rank != 4)
            {
                throw new ArgumentException(
                    $"Weight for '{opName}' must have rank 4 [C_out, C_in, K_h, K_w], got rank {weightShape.Rank}");
            }

            // Extract input dimensions
            var dimN = inputShape.GetDimension(0); // Batch size
            var dimC_in = inputShape.GetDimension(1); // Input channels
            var dimH_in = inputShape.GetDimension(2); // Input height
            var dimW_in = inputShape.GetDimension(3); // Input width

            // Extract weight dimensions
            var dimC_out = weightShape.GetDimension(0); // Output channels
            var dimC_in_weight = weightShape.GetDimension(1); // Input channels in weight
            var dimK_h = weightShape.GetDimension(2); // Kernel height
            var dimK_w = weightShape.GetDimension(3); // Kernel width

            // Validate channel dimensions match
            ValidateChannelDimension(dimC_in, dimC_in_weight, opName);

            // For simplicity in this implementation, we'll use the formula without dilation:
            // H_out = (H_in + 2*padding - K_h) / stride + 1
            // W_out = (W_in + 2*padding - K_w) / stride + 1
            // With dilation, the effective kernel size is: K_eff = K + (K - 1) * (dilation - 1)

            // Default values (can be overridden via attributes in a real implementation)
            var padding = (h: 0, w: 0); // Padding
            var stride = (h: 1, w: 1); // Stride
            var dilation = (h: 1, w: 1); // Dilation

            // Calculate effective kernel size considering dilation
            int K_h_eff = dimK_h.IsKnown() ? dimK_h.Value!.Value + (dimK_h.Value!.Value - 1) * (dilation.h - 1) : -1;
            int K_w_eff = dimK_w.IsKnown() ? dimK_w.Value!.Value + (dimK_w.Value!.Value - 1) * (dilation.w - 1) : -1;

                // Calculate output height
            SymbolicDimension dimH_out;
            if (dimH_in.IsKnown() && dimK_h.IsKnown())
            {
                int h_in = dimH_in.Value!.Value;
                int h_out = (h_in + 2 * padding.h - K_h_eff) / stride.h + 1;
                if (h_out <= 0)
                {
                    throw new ArgumentException(
                        $"Invalid output height {h_out} for '{opName}'. " +
                        $"Check padding, stride, and kernel size.");
                }
                dimH_out = SymbolicDimensionFactory.Create($"{dimH_in.Name ?? "H"}_out", h_out);
            }
            else
            {
                // Create a symbolic dimension representing the output height
                dimH_out = SymbolicDimensionFactory.Create($"{dimH_in.Name ?? "H"}_out");
            }

            // Calculate output width
            SymbolicDimension dimW_out;
            if (dimW_in.IsKnown() && dimK_w.IsKnown())
            {
                int w_in = dimW_in.Value!.Value;
                int w_out = (w_in + 2 * padding.w - K_w_eff) / stride.w + 1;
                if (w_out <= 0)
                {
                    throw new ArgumentException(
                        $"Invalid output width {w_out} for '{opName}'. " +
                        $"Check padding, stride, and kernel size.");
                }
                dimW_out = SymbolicDimensionFactory.Create($"{dimW_in.Name ?? "W"}_out", w_out);
            }
            else
            {
                // Create a symbolic dimension representing the output width
                dimW_out = SymbolicDimensionFactory.Create($"{dimW_in.Name ?? "W"}_out");
            }

            // Output shape: [N, C_out, H_out, W_out]
            return new List<SymbolicShape> { new SymbolicShape(dimN, dimC_out, dimH_out, dimW_out) };
        }

        /// <summary>
        /// Validates that input and weight channel dimensions are compatible.
        /// </summary>
        private void ValidateChannelDimension(
            SymbolicDimension inputChannels,
            SymbolicDimension weightChannels,
            string opName)
        {
            bool inputKnown = inputChannels.IsKnown();
            bool weightKnown = weightChannels.IsKnown();

            if (inputKnown && weightKnown)
            {
                int inVal = inputChannels.Value!.Value;
                int weightVal = weightChannels.Value!.Value;

                if (inVal != weightVal)
                {
                    throw new ArgumentException(
                        $"Channel dimension mismatch for '{opName}': input has {inVal} channels, " +
                        $"weight expects {weightVal} channels");
                }
            }
            // If at least one is symbolic, we assume compatibility at runtime
        }
    }
}
