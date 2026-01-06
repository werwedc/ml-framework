using MLFramework.Core;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Amp
{
    /// <summary>
    /// Registry for operation-specific precision rules in AMP
    /// </summary>
    public class AmpRegistry
    {
        private readonly Dictionary<Type, OpPrecisionRule> _rules;
        private readonly AmpConfig _config;
        private readonly object _lock = new object();

        /// <summary>
        /// Creates a new AmpRegistry with default rules
        /// </summary>
        public AmpRegistry(AmpConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _rules = new Dictionary<Type, OpPrecisionRule>();

            // Apply default rules
            DefaultAmpRules.ApplyDefaultRules(this);
        }

        /// <summary>
        /// Registers an operation to the whitelist (use lower precision)
        /// </summary>
        public void RegisterWhitelist(Type operationType, int priority = 0)
        {
            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            var rule = new OpPrecisionRule(
                operationType,
                OpPrecision.Lower,
                OpPrecision.Keep,
                priority);

            RegisterRule(rule);
        }

        /// <summary>
        /// Registers an operation to the blacklist (use higher precision)
        /// </summary>
        public void RegisterBlacklist(Type operationType, int priority = 0)
        {
            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            var rule = new OpPrecisionRule(
                operationType,
                OpPrecision.Higher,
                OpPrecision.Keep,
                priority);

            RegisterRule(rule);
        }

        /// <summary>
        /// Registers a custom precision rule for an operation
        /// </summary>
        public void RegisterCustomOp(
            Type operationType,
            DataType forwardDtype,
            DataType backwardDtype,
            int priority = 0)
        {
            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            var rule = new OpPrecisionRule(
                operationType,
                OpPrecision.Custom,
                OpPrecision.Custom,
                priority)
            {
                CustomForwardDtype = forwardDtype,
                CustomBackwardDtype = backwardDtype
            };

            RegisterRule(rule);
        }

        /// <summary>
        /// Registers a full precision rule for an operation
        /// </summary>
        public void RegisterRule(OpPrecisionRule rule)
        {
            if (rule == null)
                throw new ArgumentNullException(nameof(rule));

            lock (_lock)
            {
                // Override if new rule has higher priority, or same priority (latest wins)
                if (_rules.TryGetValue(rule.OperationType, out var existingRule))
                {
                    if (rule.Priority >= existingRule.Priority)
                    {
                        _rules[rule.OperationType] = rule;
                    }
                }
                else
                {
                    _rules[rule.OperationType] = rule;
                }
            }
        }

        /// <summary>
        /// Removes a rule for an operation
        /// </summary>
        public void Unregister(Type operationType)
        {
            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            lock (_lock)
            {
                _rules.Remove(operationType);
            }
        }

        /// <summary>
        /// Gets the precision rule for an operation
        /// </summary>
        public OpPrecisionRule? GetRule(Type operationType)
        {
            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            lock (_lock)
            {
                if (_rules.TryGetValue(operationType, out var rule))
                {
                    return rule;
                }
            }

            return null;
        }

        /// <summary>
        /// Gets the forward dtype for an operation
        /// </summary>
        public DataType GetForwardDtype(Type operationType, DataType inputDtype)
        {
            var rule = GetRule(operationType);

            if (rule != null)
            {
                return rule.GetForwardDtype(_config);
            }

            // Default policy: use input dtype
            return inputDtype;
        }

        /// <summary>
        /// Gets the backward dtype for an operation
        /// </summary>
        public DataType GetBackwardDtype(Type operationType, DataType inputDtype)
        {
            var rule = GetRule(operationType);

            if (rule != null)
            {
                return rule.GetBackwardDtype(_config);
            }

            // Default policy: use input dtype
            return inputDtype;
        }

        /// <summary>
        /// Checks if an operation is in the whitelist
        /// </summary>
        public bool IsWhitelisted(Type operationType)
        {
            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            var rule = GetRule(operationType);
            return rule != null && rule.ForwardPrecision == OpPrecision.Lower;
        }

        /// <summary>
        /// Checks if an operation is in the blacklist
        /// </summary>
        public bool IsBlacklisted(Type operationType)
        {
            if (operationType == null)
                throw new ArgumentNullException(nameof(operationType));

            var rule = GetRule(operationType);
            return rule != null && rule.ForwardPrecision == OpPrecision.Higher;
        }

        /// <summary>
        /// Clears all registered rules
        /// </summary>
        public void Clear()
        {
            lock (_lock)
            {
                _rules.Clear();
            }
        }

        /// <summary>
        /// Gets all registered rules
        /// </summary>
        public IReadOnlyDictionary<Type, OpPrecisionRule> GetAllRules()
        {
            lock (_lock)
            {
                return new Dictionary<Type, OpPrecisionRule>(_rules);
            }
        }

        /// <summary>
        /// Gets the AMP configuration
        /// </summary>
        public AmpConfig GetConfig()
        {
            return _config;
        }
    }
}
