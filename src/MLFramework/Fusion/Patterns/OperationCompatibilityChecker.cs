namespace MLFramework.Fusion
{
    /// <summary>
    /// Checks compatibility between operations for fusion
    /// </summary>
    public class OperationCompatibilityChecker
    {
        /// <summary>
        /// Checks if two operations can be fused together
        /// </summary>
        public bool CanFuse(Operation op1, Operation op2)
        {
            return CheckMemoryLayout(op1, op2) &&
                   CheckNumericalPrecision(op1, op2) &&
                   CheckThreadBlockConfig(op1, op2) &&
                   CheckSideEffects(op1, op2) &&
                   CheckControlFlow(op1, op2);
        }

        /// <summary>
        /// Checks if two operations use compatible memory layouts
        /// </summary>
        public bool CheckMemoryLayout(Operation op1, Operation op2)
        {
            var layout1 = op1.Layout;
            var layout2 = op2.Layout;

            // If either layout is "Any", they're compatible
            if (layout1 == TensorLayout.Any || layout2 == TensorLayout.Any)
                return true;

            // Layouts must match
            return layout1 == layout2;
        }

        /// <summary>
        /// Checks if two operations use compatible numerical precision
        /// </summary>
        public bool CheckNumericalPrecision(Operation op1, Operation op2)
        {
            var dtype1 = op1.DataType;
            var dtype2 = op2.DataType;

            // Data types must match for fusion
            return dtype1 == dtype2;
        }

        /// <summary>
        /// Checks if two operations have compatible thread block configurations
        /// </summary>
        public bool CheckThreadBlockConfig(Operation op1, Operation op2)
        {
            var config1 = op1.GetThreadBlockConfig();
            var config2 = op2.GetThreadBlockConfig();

            // If either has no config, they're compatible
            if (config1 == null || config2 == null)
                return true;

            // Thread block configurations must match
            return config1.X == config2.X &&
                   config1.Y == config2.Y &&
                   config1.Z == config2.Z;
        }

        /// <summary>
        /// Checks if operations have side effects
        /// </summary>
        public bool CheckSideEffects(Operation op1, Operation op2)
        {
            // Neither operation should have side effects
            return !HasSideEffects(op1) && !HasSideEffects(op2);
        }

        /// <summary>
        /// Checks if operations have complex control flow
        /// </summary>
        public bool CheckControlFlow(Operation op1, Operation op2)
        {
            // Neither operation should have data-dependent control flow
            return !HasDataDependentControlFlow(op1) && !HasDataDependentControlFlow(op2);
        }

        /// <summary>
        /// Determines if operation has side effects
        /// </summary>
        private bool HasSideEffects(Operation op)
        {
            return op.Type switch
            {
                "Print" or "WriteToFile" or "Send" or
                "Log" or "DebugPrint"
                    => true,

                _ => false
            };
        }

        /// <summary>
        /// Determines if operation has data-dependent control flow
        /// </summary>
        private bool HasDataDependentControlFlow(Operation op)
        {
            return op.Type switch
            {
                "Where" or "DynamicIf" or "Conditional" or
                "Switch" or "MaskedSelect"
                    => true,

                _ => false
            };
        }

        /// <summary>
        /// Checks if multiple operations in a sequence are compatible
        /// </summary>
        public bool AreSequenceCompatible(IReadOnlyList<Operation> operations)
        {
            if (operations.Count < 2)
                return true;

            for (int i = 0; i < operations.Count - 1; i++)
            {
                if (!CanFuse(operations[i], operations[i + 1]))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets incompatibility reason if operations cannot be fused
        /// </summary>
        public string? GetIncompatibilityReason(Operation op1, Operation op2)
        {
            if (!CheckMemoryLayout(op1, op2))
                return $"Memory layout mismatch: {op1.Layout} vs {op2.Layout}";

            if (!CheckNumericalPrecision(op1, op2))
                return $"Data type mismatch: {op1.DataType} vs {op2.DataType}";

            if (!CheckThreadBlockConfig(op1, op2))
                return "Thread block configuration mismatch";

            if (!CheckSideEffects(op1, op2))
            {
                var opWithSideEffect = HasSideEffects(op1) ? op1 : op2;
                return $"Operation {opWithSideEffect.Name} ({opWithSideEffect.Type}) has side effects";
            }

            if (!CheckControlFlow(op1, op2))
            {
                var opWithControlFlow = HasDataDependentControlFlow(op1) ? op1 : op2;
                return $"Operation {opWithControlFlow.Name} ({opWithControlFlow.Type}) has data-dependent control flow";
            }

            return null;
        }
    }
}
