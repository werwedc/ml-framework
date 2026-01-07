namespace MLFramework.Fusion
{
    /// <summary>
    /// Detects Conv/Linear -> Activation patterns for fusion
    /// </summary>
    public class ConvActivationPatternDetector : IFusionPatternDetector
    {
        private readonly GraphAnalyzer _graphAnalyzer;
        private readonly OperationCompatibilityChecker _compatibilityChecker;

        public ConvActivationPatternDetector()
        {
            _graphAnalyzer = new GraphAnalyzer();
            _compatibilityChecker = new OperationCompatibilityChecker();
        }

        /// <summary>
        /// Detects conv -> activation patterns in computational graph
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
                        PatternType = FusionPatternType.ConvActivation,
                        BenefitScore = benefitScore
                    });
                }
            }

            return candidates;
        }

        /// <summary>
        /// Checks if operations form a conv/linear -> activation pattern
        /// </summary>
        public bool IsFusible(IEnumerable<Operation> operations)
        {
            var opList = operations.ToList();

            if (opList.Count != 2)
                return false;

            var firstOp = opList[0];
            var secondOp = opList[1];

            // Check pattern: conv/linear -> activation
            if (!IsConvOrLinear(firstOp.Type))
                return false;

            if (!IsActivationFunction(secondOp.Type))
                return false;

            // Check compatibility
            if (!_compatibilityChecker.CanFuse(firstOp, secondOp))
                return false;

            // Check parameter compatibility
            if (!CheckParameterCompatibility(firstOp, secondOp))
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
        /// Determines if operation is an activation function
        /// </summary>
        private bool IsActivationFunction(string opType)
        {
            return opType switch
            {
                "ReLU" or "LeakyReLU" or "Sigmoid" or
                "Tanh" or "ELU" or "SELU" or "GELU" or
                "Swish" or "Mish" or "HardSwish"
                    => true,

                _ => false
            };
        }

        /// <summary>
        /// Checks if convolution and activation have compatible parameters
        /// </summary>
        private bool CheckParameterCompatibility(Operation convOp, Operation activationOp)
        {
            // For standard activations (ReLU, Sigmoid, Tanh), no special parameters needed
            if (activationOp.Type == "ReLU" || activationOp.Type == "Sigmoid" || activationOp.Type == "Tanh")
                return true;

            // For LeakyReLU, check if alpha parameter is present and valid
            if (activationOp.Type == "LeakyReLU")
            {
                if (activationOp.Attributes.TryGetValue("alpha", out var alphaObj))
                {
                    if (alphaObj is float alpha && alpha >= 0.0f)
                        return true;
                }
                // Default alpha of 0.01 is acceptable
                return true;
            }

            // For ELU, check alpha parameter
            if (activationOp.Type == "ELU")
            {
                if (activationOp.Attributes.TryGetValue("alpha", out var alphaObj))
                {
                    if (alphaObj is float alpha && alpha >= 0.0f)
                        return true;
                }
                return true;
            }

            // Other activations are acceptable as-is
            return true;
        }

        /// <summary>
        /// Computes benefit score for conv-activation fusion
        /// </summary>
        private int ComputeBenefitScore(List<Operation> operations)
        {
            var firstOp = operations[0];

            // Base score for removing activation kernel launch
            var baseScore = 30;

            // Additional score based on convolution size
            var outputElements = firstOp.OutputShape.Size;
            var sizeBonus = (int)Math.Log10(outputElements) * 10;

            // Bonus for specific convolution types
            var typeBonus = firstOp.Type switch
            {
                "Conv2D" or "Conv3D" => 20,
                "Linear" => 15,
                _ => 10
            };

            return baseScore + sizeBonus + typeBonus;
        }
    }
}
