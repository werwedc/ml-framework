using MLFramework.Core;

namespace MLFramework.Fusion
{
    /// <summary>
    /// Detects fusible operation sequences in computational graphs
    /// </summary>
    public interface IFusionPatternDetector
    {
        /// <summary>
        /// Detects fusible operation sequences in computational graph
        /// </summary>
        /// <param name="graph">Computational graph to analyze</param>
        /// <returns>List of detected fusion candidate sequences</returns>
        List<FusionCandidate> DetectPatterns(ComputationalGraph graph);

        /// <summary>
        /// Checks if a specific operation sequence is fusible
        /// </summary>
        bool IsFusible(IEnumerable<Operation> operations);
    }

    /// <summary>
    /// Candidate sequence of operations that can be fused together
    /// </summary>
    public record FusionCandidate
    {
        public required IReadOnlyList<Operation> Operations { get; init; }
        public required FusionPatternType PatternType { get; init; }
        public required int BenefitScore { get; init; }
    }

    /// <summary>
    /// Types of fusion patterns supported
    /// </summary>
    public enum FusionPatternType
    {
        /// <summary>
        /// Sequence of element-wise operations (e.g., Add -> Mul -> ReLU)
        /// </summary>
        ElementWise,

        /// <summary>
        /// Reduction operation followed by element-wise operation
        /// </summary>
        ReductionThenElementWise,

        /// <summary>
        /// Convolution followed by activation function
        /// </summary>
        ConvActivation,

        /// <summary>
        /// Convolution followed by batch normalization (for weight folding)
        /// </summary>
        ConvBatchNorm,

        /// <summary>
        /// Linear/dense layer followed by activation function
        /// </summary>
        LinearActivation,

        /// <summary>
        /// Mixed pattern with various operation types
        /// </summary>
        Mixed
    }

    /// <summary>
    /// Computational graph representation for fusion analysis
    /// </summary>
    public class ComputationalGraph
    {
        public required string Id { get; init; }
        public required IReadOnlyList<Operation> Operations { get; init; }
        public required DependencyGraph DependencyGraph { get; init; }
    }

    /// <summary>
    /// Helper methods for Operation access
    /// </summary>
    public static class OperationExtensions
    {
        /// <summary>
        /// Gets thread block configuration from operation attributes
        /// </summary>
        public static Backends.ThreadBlockConfiguration? GetThreadBlockConfig(this Operation op)
        {
            if (op.Attributes.TryGetValue("threadBlockConfig", out var config) && config is Backends.ThreadBlockConfiguration tbc)
                return tbc;
            return null;
        }

        /// <summary>
        /// Gets dependency list from operation inputs
        /// </summary>
        public static IReadOnlyList<string> GetDependencies(this Operation op)
        {
            return op.Inputs;
        }
    }
}
