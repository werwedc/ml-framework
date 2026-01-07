using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.IR.Transformations
{
    using MLFramework.IR.Graph;

    /// <summary>
    /// Manager for running IR transformation passes in a specified order
    /// </summary>
    public class IRPassManager
    {
        /// <summary>
        /// Types of transformation passes
        /// </summary>
        public enum PassType
        {
            /// <summary>Analysis passes that don't modify the IR</summary>
            Analysis,
            /// <summary>Optimization passes that improve the IR</summary>
            Optimization,
            /// <summary>Lowering passes that transform IR between levels</summary>
            Lowering,
            /// <summary>Validation passes that check IR correctness</summary>
            Validation
        }

        private readonly List<IRTransformation> _passes;
        private readonly Dictionary<PassType, List<IRTransformation>> _passesByType;

        /// <summary>
        /// Gets the total number of registered passes
        /// </summary>
        public int PassCount => _passes.Count;

        /// <summary>
        /// Initializes a new IRPassManager instance
        /// </summary>
        public IRPassManager()
        {
            _passes = new List<IRTransformation>();
            _passesByType = new Dictionary<PassType, List<IRTransformation>>();

            foreach (PassType type in Enum.GetValues(typeof(PassType)))
            {
                _passesByType[type] = new List<IRTransformation>();
            }
        }

        /// <summary>
        /// Adds a pass to the manager
        /// </summary>
        /// <param name="pass">The pass to add</param>
        /// <param name="type">The type of pass</param>
        public void AddPass(IRTransformation pass, PassType type = PassType.Optimization)
        {
            if (pass == null)
                throw new ArgumentNullException(nameof(pass));

            _passes.Add(pass);
            _passesByType[type].Add(pass);
        }

        /// <summary>
        /// Runs all registered passes on the module
        /// </summary>
        /// <param name="module">The module to transform</param>
        /// <returns>True if any pass modified the module, false otherwise</returns>
        public bool RunAll(Graph.HLIRModule module)
        {
            bool changed = false;
            foreach (var pass in _passes)
            {
                pass.Initialize(module);
                try
                {
                    bool passChanged = pass.Run(module);
                    changed |= passChanged;
                }
                finally
                {
                    pass.Cleanup();
                }
            }
            return changed;
        }

        /// <summary>
        /// Runs only analysis passes on the module
        /// </summary>
        /// <param name="module">The module to analyze</param>
        /// <returns>True if any analysis pass reported issues, false otherwise</returns>
        public bool RunAnalysisPasses(Graph.HLIRModule module)
        {
            bool changed = false;
            foreach (var pass in _passesByType[PassType.Analysis])
            {
                pass.Initialize(module);
                try
                {
                    bool passChanged = pass.Run(module);
                    changed |= passChanged;
                }
                finally
                {
                    pass.Cleanup();
                }
            }
            return changed;
        }

        /// <summary>
        /// Runs only optimization passes on the module
        /// </summary>
        /// <param name="module">The module to optimize</param>
        /// <returns>True if any optimization pass modified the module, false otherwise</returns>
        public bool RunOptimizationPasses(Graph.HLIRModule module)
        {
            bool changed = false;
            foreach (var pass in _passesByType[PassType.Optimization])
            {
                pass.Initialize(module);
                try
                {
                    bool passChanged = pass.Run(module);
                    changed |= passChanged;
                }
                finally
                {
                    pass.Cleanup();
                }
            }
            return changed;
        }

        /// <summary>
        /// Runs only lowering passes on the module
        /// </summary>
        /// <param name="module">The module to lower</param>
        /// <returns>True if any lowering pass modified the module, false otherwise</returns>
        public bool RunLoweringPasses(Graph.HLIRModule module)
        {
            bool changed = false;
            foreach (var pass in _passesByType[PassType.Lowering])
            {
                pass.Initialize(module);
                try
                {
                    bool passChanged = pass.Run(module);
                    changed |= passChanged;
                }
                finally
                {
                    pass.Cleanup();
                }
            }
            return changed;
        }

        /// <summary>
        /// Runs only validation passes on the module
        /// </summary>
        /// <param name="module">The module to validate</param>
        /// <returns>True if all validation passes succeeded, false if any failed</returns>
        public bool RunValidationPasses(Graph.HLIRModule module)
        {
            bool allValid = true;
            foreach (var pass in _passesByType[PassType.Validation])
            {
                pass.Initialize(module);
                try
                {
                    bool passValid = pass.Run(module);
                    allValid &= passValid;
                }
                finally
                {
                    pass.Cleanup();
                }
            }
            return allValid;
        }

        /// <summary>
        /// Gets all passes of a specific type
        /// </summary>
        /// <param name="type">The type of passes to get</param>
        /// <returns>A list of passes of the specified type</returns>
        public IReadOnlyList<IRTransformation> GetPasses(PassType type)
        {
            return _passesByType[type].AsReadOnly();
        }

        /// <summary>
        /// Clears all registered passes
        /// </summary>
        public void Clear()
        {
            _passes.Clear();
            foreach (var type in Enum.GetValues(typeof(PassType)).Cast<PassType>())
            {
                _passesByType[type].Clear();
            }
        }
    }
}
