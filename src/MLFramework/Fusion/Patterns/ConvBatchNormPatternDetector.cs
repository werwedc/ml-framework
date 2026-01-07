namespace MLFramework.Fusion
{
    /// <summary>
    /// Detects Conv/Linear -> BatchNorm patterns for weight folding
    /// </summary>
    public class ConvBatchNormPatternDetector : IFusionPatternDetector
    {
        private readonly GraphAnalyzer _graphAnalyzer;
        private readonly OperationCompatibilityChecker _compatibilityChecker;

        public ConvBatchNormPatternDetector()
        {
            _graphAnalyzer = new GraphAnalyzer();
            _compatibilityChecker = new OperationCompatibilityChecker();
        }

        /// <summary>
        /// Detects conv -> batchnorm patterns in computational graph
        /// </summary>
        public List<FusionCandidate> DetectPatterns(ComputationalGraph graph)
        {
            var candidates = new List<FusionCandidate>();
            var chains = _graphAnalyzer.FindLinearChains(graph.DependencyGraph);

            foreach (var chain in chains)
            {
                if (chain.Operations.Count == 2 && IsFusible(chain.Operations))
                {
                    var opList = chain.Operations.ToList();
                    var benefitScore = ComputeBenefitScore(opList);

                    candidates.Add(new FusionCandidate
                    {
                        Operations = opList,
                        PatternType = FusionPatternType.ConvBatchNorm,
                        BenefitScore = benefitScore
                    });
                }
            }

            return candidates;
        }

        /// <summary>
        /// Checks if operations form a conv/linear -> batchnorm pattern
        /// </summary>
        public bool IsFusible(IEnumerable<Operation> operations)
        {
            var opList = operations.ToList();

            if (opList.Count != 2)
                return false;

            var firstOp = opList[0];
            var secondOp = opList[1];

            // Check pattern: conv/linear -> batchnorm
            if (!IsConvOrLinear(firstOp.Type))
                return false;

            if (secondOp.Type != "BatchNorm")
                return false;

            // Check compatibility
            if (!_compatibilityChecker.CanFuse(firstOp, secondOp))
                return false;

            // Validate batchnorm parameters for folding
            if (!ValidateBatchNormForFolding(secondOp))
                return false;

            return true;
        }

        /// <summary>
        /// Determines if operation is a convolution or linear layer
        /// </summary>
        private bool IsConvOrLinear(string opType)
        {
            return opType switch
            {
                "Conv1D" or "Conv2D" or "Conv3D" or
                "ConvTranspose1D" or "ConvTranspose2D" or "ConvTranspose3D" or
                "Linear" or "Dense" or "MatMul"
                    => true,

                _ => false
            };
        }

        /// <summary>
        /// Validates that batchnorm parameters allow folding
        /// </summary>
        private bool ValidateBatchNormForFolding(Operation batchNormOp)
        {
            // Check if training mode - batchnorm can only be folded during inference
            if (batchNormOp.Attributes.TryGetValue("training", out var trainingObj))
            {
                if (trainingObj is bool training && training)
                {
                    // BatchNorm in training mode cannot be folded
                    return false;
                }
            }

            // Check for required parameters
            var requiredParams = new[] { "gamma", "beta", "running_mean", "running_var" };

            foreach (var param in requiredParams)
            {
                if (!batchNormOp.Attributes.ContainsKey(param))
                    return false;
            }

            // Validate epsilon parameter (should be positive)
            if (batchNormOp.Attributes.TryGetValue("epsilon", out var epsilonObj))
            {
                if (epsilonObj is float epsilon && epsilon <= 0.0f)
                    return false;
            }

            // Validate momentum parameter (should be between 0 and 1)
            if (batchNormOp.Attributes.TryGetValue("momentum", out var momentumObj))
            {
                if (momentumObj is float momentum && (momentum < 0.0f || momentum > 1.0f))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Computes benefit score for conv-batchnorm folding
        /// </summary>
        private int ComputeBenefitScore(List<Operation> operations)
        {
            var firstOp = operations[0];

            // Base score for removing batchnorm kernel launch
            var baseScore = 25;

            // Additional score based on convolution size
            var outputElements = firstOp.OutputShape.Size;
            var sizeBonus = (int)Math.Log10(outputElements) * 10;

            // Bonus for specific convolution types
            var typeBonus = firstOp.Type switch
            {
                "Conv2D" or "Conv3D" => 25,
                "Conv1D" => 20,
                "Linear" => 15,
                _ => 10
            };

            return baseScore + sizeBonus + typeBonus;
        }
    }
}
