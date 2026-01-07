namespace MLFramework.Fusion
{
    /// <summary>
    /// Detects chains of element-wise operations that can be fused
    /// </summary>
    public class ElementWisePatternDetector : IFusionPatternDetector
    {
        private readonly GraphAnalyzer _graphAnalyzer;

        public ElementWisePatternDetector()
        {
            _graphAnalyzer = new GraphAnalyzer();
        }

        /// <summary>
        /// Detects element-wise operation chains in computational graph
        /// </summary>
        public List<FusionCandidate> DetectPatterns(ComputationalGraph graph)
        {
            var candidates = new List<FusionCandidate>();
            var chains = _graphAnalyzer.FindLinearChains(graph.DependencyGraph);

            foreach (var chain in chains)
            {
                if (IsFusible(chain.Operations))
                {
                    var opList = chain.Operations.ToList();
                    var benefitScore = ComputeBenefitScore(opList);

                    candidates.Add(new FusionCandidate
                    {
                        Operations = opList,
                        PatternType = FusionPatternType.ElementWise,
                        BenefitScore = benefitScore
                    });
                }
            }

            return candidates;
        }

        /// <summary>
        /// Checks if all operations in sequence are element-wise and compatible
        /// </summary>
        public bool IsFusible(IEnumerable<Operation> operations)
        {
            var opList = operations.ToList();

            if (opList.Count < 2)
                return false;

            // Check all operations are element-wise
            foreach (var op in opList)
            {
                if (!IsElementWiseOperation(op.Type))
                    return false;
            }

            // Check memory layout compatibility
            if (!CheckMemoryLayoutCompatibility(opList))
                return false;

            // Check data type compatibility
            if (!CheckDataTypeCompatibility(opList))
                return false;

            // Check shape compatibility
            if (!CheckShapeCompatibility(opList))
                return false;

            return true;
        }

        /// <summary>
        /// Determines if operation type is element-wise
        /// </summary>
        private bool IsElementWiseOperation(string opType)
        {
            return opType switch
            {
                "Add" or "Sub" or "Mul" or "Div" or
                "ReLU" or "Sigmoid" or "Tanh" or "LeakyReLU" or
                "Exp" or "Log" or "Sqrt" or "Abs" or
                "Pow" or "Sin" or "Cos" or "Neg"
                    => true,

                _ => false
            };
        }

        /// <summary>
        /// Checks if all operations use compatible memory layouts
        /// </summary>
        private bool CheckMemoryLayoutCompatibility(List<Operation> operations)
        {
            if (operations.Count == 0)
                return true;

            var referenceLayout = operations[0].Layout;

            for (int i = 1; i < operations.Count; i++)
            {
                var opLayout = operations[i].Layout;

                if (opLayout != TensorLayout.Any &&
                    referenceLayout != TensorLayout.Any &&
                    opLayout != referenceLayout)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Checks if all operations use compatible data types
        /// </summary>
        private bool CheckDataTypeCompatibility(List<Operation> operations)
        {
            if (operations.Count == 0)
                return true;

            var referenceType = operations[0].DataType;

            for (int i = 1; i < operations.Count; i++)
            {
                if (operations[i].DataType != referenceType)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Checks if operations have compatible shapes
        /// </summary>
        private bool CheckShapeCompatibility(List<Operation> operations)
        {
            if (operations.Count == 0)
                return true;

            // For element-wise operations, all shapes should be equal
            var referenceShape = operations[0].OutputShape;

            for (int i = 1; i < operations.Count; i++)
            {
                var shape = operations[i].OutputShape;

                if (shape.Size != referenceShape.Size)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Computes benefit score for fusing this chain
        /// </summary>
        private int ComputeBenefitScore(List<Operation> operations)
        {
            // Base score for each operation in chain
            var baseScore = operations.Count * 10;

            // Bonus for longer chains (memory reduction)
            var lengthBonus = (int)Math.Pow(operations.Count, 1.5);

            // Bonus for operations that require global memory access
            var memoryAccessBonus = operations.Count * 5;

            return baseScore + lengthBonus + memoryAccessBonus;
        }
    }
}
