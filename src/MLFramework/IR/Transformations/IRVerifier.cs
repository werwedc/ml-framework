using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.IR.Operations;
using MLFramework.IR.Values;

namespace MLFramework.IR.Transformations
{
    /// <summary>
    /// Verifier for checking IR correctness and validity
    /// </summary>
    public class IRVerifier : IRTransformation
    {
        private List<string> _errors;
        private List<string> _warnings;

        /// <summary>
        /// Gets the list of errors found during verification
        /// </summary>
        public List<string> Errors => _errors;

        /// <summary>
        /// Gets the list of warnings found during verification
        /// </summary>
        public List<string> Warnings => _warnings;

        /// <summary>
        /// Gets the total number of errors
        /// </summary>
        public int ErrorCount => _errors.Count;

        /// <summary>
        /// Gets the total number of warnings
        /// </summary>
        public int WarningCount => _warnings.Count;

        /// <summary>
        /// Initializes a new IRVerifier instance
        /// </summary>
        public IRVerifier()
            : base("Verifier", true)
        {
            _errors = new List<string>();
            _warnings = new List<string>();
        }

        /// <summary>
        /// Runs the verifier on the given module
        /// </summary>
        /// <param name="module">The module to verify</param>
        /// <returns>True if verification succeeded (no errors), false otherwise</returns>
        public override bool Run(HLIRModule module)
        {
            _errors.Clear();
            _warnings.Clear();

            if (module == null)
            {
                _errors.Add("Module is null");
                return false;
            }

            try
            {
                // Verify each function in the module
                foreach (var function in module.Functions)
                {
                    VerifyFunction(function);
                }

                // Verify module-level constants
                VerifyConstants(module);
            }
            catch (Exception ex)
            {
                _errors.Add($"Exception during verification: {ex.Message}");
                return false;
            }

            return _errors.Count == 0;
        }

        /// <summary>
        /// Verifies a single function
        /// </summary>
        /// <param name="function">The function to verify</param>
        private void VerifyFunction(HIRFunction function)
        {
            if (function == null)
            {
                _errors.Add("Function is null");
                return;
            }

            // Verify function has a name
            if (string.IsNullOrEmpty(function.Name))
            {
                _errors.Add($"Function has no name (ID: {function.GetHashCode()})");
            }

            // Verify function has a body
            if (function.Body == null)
            {
                _errors.Add($"Function '{function.Name}' has no body");
                return;
            }

            // Verify parameters
            VerifyParameters(function);

            // Verify results
            VerifyResults(function);

            // Verify body blocks
            VerifyBlock(function.Body, function.Name, "body");
        }

        /// <summary>
        /// Verifies function parameters
        /// </summary>
        /// <param name="function">The function whose parameters to verify</param>
        private void VerifyParameters(HIRFunction function)
        {
            foreach (var param in function.Parameters)
            {
                if (param == null)
                {
                    _errors.Add($"Function '{function.Name}' has null parameter");
                    continue;
                }

                if (param.Type == null)
                {
                    _errors.Add($"Function '{function.Name}' parameter '{param.Name}' has no type");
                }
            }
        }

        /// <summary>
        /// Verifies function results
        /// </summary>
        /// <param name="function">The function whose results to verify</param>
        private void VerifyResults(HIRFunction function)
        {
            foreach (var result in function.Results)
            {
                if (result == null)
                {
                    _errors.Add($"Function '{function.Name}' has null result");
                    continue;
                }

                if (result.Type == null)
                {
                    _errors.Add($"Function '{function.Name}' result '{result.Name}' has no type");
                }
            }
        }

        /// <summary>
        /// Verifies a single block
        /// </summary>
        /// <param name="block">The block to verify</param>
        /// <param name="functionName">The name of the function containing the block</param>
        /// <param name="blockContext">Context description of the block</param>
        private void VerifyBlock(IRBlock block, string functionName, string blockContext)
        {
            if (block == null)
            {
                _errors.Add($"Function '{functionName}' has null block ({blockContext})");
                return;
            }

            // Track all defined values in this block
            var definedValues = new HashSet<int>();

            // Add block arguments as defined values
            foreach (var arg in block.Arguments)
            {
                if (arg == null)
                {
                    _errors.Add($"Function '{function.Name}' block '{block.Name}' has null argument");
                    continue;
                }

                if (!definedValues.Add(arg.Id))
                {
                    _errors.Add($"Function '{function.Name}' block '{block.Name}' has duplicate argument ID: {arg.Id}");
                }

                if (arg.Type == null)
                {
                    _errors.Add($"Function '{function.Name}' block '{block.Name}' argument '{arg.Name}' has no type");
                }
            }

            // Verify all operations
            foreach (var op in block.Operations)
            {
                VerifyOperation(op, functionName, block.Name, definedValues);
            }

            // Verify return values
            foreach (var ret in block.Returns)
            {
                if (ret == null)
                {
                    _errors.Add($"Function '{function.Name}' block '{block.Name}' has null return value");
                    continue;
                }

                if (!definedValues.Contains(ret.Id))
                {
                    _errors.Add($"Function '{function.Name}' block '{block.Name}' return value {ret.Id} is not defined");
                }
            }
        }

        /// <summary>
        /// Verifies a single operation
        /// </summary>
        /// <param name="op">The operation to verify</param>
        /// <param name="functionName">The name of the function containing the operation</param>
        /// <param name="blockName">The name of the block containing the operation</param>
        /// <param name="definedValues">Set of defined value IDs in the block</param>
        private void VerifyOperation(IROperation op, string functionName, string blockName, HashSet<int> definedValues)
        {
            if (op == null)
            {
                _errors.Add($"Function '{functionName}' block '{blockName}' has null operation");
                return;
            }

            // Verify operands are defined before use
            foreach (var operand in op.Operands)
            {
                if (operand == null)
                {
                    _errors.Add($"Function '{functionName}' block '{blockName}' operation '{op.Name}' has null operand");
                    continue;
                }

                if (!definedValues.Contains(operand.Id))
                {
                    _errors.Add($"Function '{functionName}' block '{blockName}' operation '{op.Name}' operand {operand.Id} is used before being defined");
                }

                if (operand.Type == null)
                {
                    _warnings.Add($"Function '{functionName}' block '{blockName}' operation '{op.Name}' operand '{operand.Name}' has no type");
                }
            }

            // Verify results
            foreach (var result in op.Results)
            {
                if (result == null)
                {
                    _errors.Add($"Function '{functionName}' block '{blockName}' operation '{op.Name}' has null result");
                    continue;
                }

                if (!definedValues.Add(result.Id))
                {
                    _errors.Add($"Function '{functionName}' block '{blockName}' operation '{op.Name}' produces duplicate result ID: {result.Id}");
                }

                if (result.Type == null)
                {
                    _warnings.Add($"Function '{functionName}' block '{blockName}' operation '{op.Name}' result '{result.Name}' has no type");
                }
            }

            // Try to validate the operation itself (if it has a Validate method)
            try
            {
                var validateMethod = op.GetType().GetMethod("Validate");
                if (validateMethod != null && validateMethod.GetParameters().Length == 0)
                {
                    validateMethod.Invoke(op, null);
                }
            }
            catch (Exception ex)
            {
                _errors.Add($"Function '{functionName}' block '{blockName}' operation '{op.Name}' validation failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Verifies module-level constants
        /// </summary>
        /// <param name="module">The module whose constants to verify</param>
        private void VerifyConstants(HLIRModule module)
        {
            foreach (var (name, value) in module.Constants)
            {
                if (string.IsNullOrEmpty(name))
                {
                    _warnings.Add("Module has constant with empty name");
                }

                if (value == null)
                {
                    _errors.Add($"Module constant '{name}' is null");
                }
            }
        }

        /// <summary>
        /// Gets a formatted error message
        /// </summary>
        /// <returns>A string containing all errors</returns>
        public string GetErrorMessage()
        {
            if (_errors.Count == 0)
                return "No errors found";

            return $"Found {_errors.Count} error(s):\n" + string.Join("\n", _errors.Select((e, i) => $"  {i + 1}. {e}"));
        }

        /// <summary>
        /// Gets a formatted warning message
        /// </summary>
        /// <returns>A string containing all warnings</returns>
        public string GetWarningMessage()
        {
            if (_warnings.Count == 0)
                return "No warnings found";

            return $"Found {_warnings.Count} warning(s):\n" + string.Join("\n", _warnings.Select((w, i) => $"  {i + 1}. {w}"));
        }

        /// <summary>
        /// Gets a complete formatted message including both errors and warnings
        /// </summary>
        /// <returns>A string containing all errors and warnings</returns>
        public string GetFullReport()
        {
            var report = new List<string>();

            if (_errors.Count > 0)
            {
                report.Add(GetErrorMessage());
            }

            if (_warnings.Count > 0)
            {
                report.Add(GetWarningMessage());
            }

            if (_errors.Count == 0 && _warnings.Count == 0)
            {
                return "Verification passed: No errors or warnings found";
            }

            return string.Join("\n\n", report);
        }
    }
}
