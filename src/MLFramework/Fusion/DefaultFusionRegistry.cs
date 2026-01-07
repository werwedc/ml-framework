using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Default implementation of fusion registry with pre-registered common patterns
/// </summary>
public class DefaultFusionRegistry : IFusionRegistry
{
    private readonly Dictionary<string, FusibleOpConstraints> _fusibleOps = new();
    private readonly Dictionary<string, FusionPatternDefinition> _patterns = new();
    private readonly List<FusionPatternDefinition> _orderedPatterns = new();

    /// <summary>
    /// Initializes a new instance of DefaultFusionRegistry
    /// </summary>
    /// <param name="skipDefaults">If true, skip registering default patterns</param>
    public DefaultFusionRegistry(bool skipDefaults = false)
    {
        if (!skipDefaults)
        {
            RegisterDefaultElementWiseOperations();
            RegisterDefaultFusionPatterns();
        }
    }

    /// <inheritdoc/>
    public void RegisterFusibleOperation(string opType, FusibleOpConstraints constraints)
    {
        ArgumentNullException.ThrowIfNull(opType);
        ArgumentNullException.ThrowIfNull(constraints);

        _fusibleOps[opType] = constraints;
    }

    /// <inheritdoc/>
    public void RegisterFusionPattern(string patternName, FusionPatternDefinition pattern)
    {
        ArgumentNullException.ThrowIfNull(patternName);
        ArgumentNullException.ThrowIfNull(pattern);

        _patterns[patternName] = pattern;

        // Add to ordered list and maintain priority order
        _orderedPatterns.Add(pattern);
        _orderedPatterns.Sort((a, b) => b.Priority.CompareTo(a.Priority));
    }

    /// <inheritdoc/>
    public IReadOnlySet<string> GetFusibleOperations()
    {
        return _fusibleOps.Keys.ToHashSet();
    }

    /// <inheritdoc/>
    public FusionPatternDefinition? GetPattern(string patternName)
    {
        ArgumentNullException.ThrowIfNull(patternName);

        return _patterns.TryGetValue(patternName, out var pattern) ? pattern : null;
    }

    /// <inheritdoc/>
    public List<FusionPatternMatch> FindMatches(IEnumerable<Operation> operations)
    {
        ArgumentNullException.ThrowIfNull(operations);

        var opList = operations.ToList();
        var matches = new List<FusionPatternMatch>();

        foreach (var pattern in _orderedPatterns)
        {
            if (pattern.MatchStrategy(opList))
            {
                // Calculate match score based on priority and operation count
                int matchScore = CalculateMatchScore(pattern, opList);
                matches.Add(new FusionPatternMatch
                {
                    Pattern = pattern,
                    MatchedOperations = opList,
                    MatchScore = matchScore
                });
            }
        }

        // Return matches sorted by score
        return matches.OrderByDescending(m => m.MatchScore).ToList();
    }

    private int CalculateMatchScore(FusionPatternDefinition pattern, IReadOnlyList<Operation> operations)
    {
        // Base score from pattern priority
        int score = pattern.Priority * 100;

        // Bonus for matching more operations
        score += operations.Count * 10;

        return score;
    }

    private void RegisterDefaultElementWiseOperations()
    {
        // Register element-wise operations
        RegisterFusibleOperation("Add", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16,
                DataType.Int32, DataType.Int64
            },
            SupportsFusionWithInplaceOps = true
        });

        RegisterFusibleOperation("Mul", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16,
                DataType.Int32, DataType.Int64
            },
            SupportsFusionWithInplaceOps = true
        });

        RegisterFusibleOperation("Sub", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16,
                DataType.Int32, DataType.Int64
            },
            SupportsFusionWithInplaceOps = true
        });

        RegisterFusibleOperation("Div", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16,
                DataType.Int32, DataType.Int64
            },
            SupportsFusionWithInplaceOps = false
        });

        RegisterFusibleOperation("ReLU", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16
            }
        });

        RegisterFusibleOperation("Sigmoid", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16
            }
        });

        RegisterFusibleOperation("Tanh", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16
            }
        });

        RegisterFusibleOperation("LeakyReLU", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16
            }
        });

        RegisterFusibleOperation("Exp", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16
            }
        });

        RegisterFusibleOperation("Log", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16
            }
        });

        RegisterFusibleOperation("Abs", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16,
                DataType.Int32, DataType.Int64
            }
        });

        RegisterFusibleOperation("Neg", new FusibleOpConstraints
        {
            RequiredLayout = TensorLayout.Any,
            SupportedDataTypes = new HashSet<DataType> {
                DataType.Float32, DataType.Float16,
                DataType.Int32, DataType.Int64
            }
        });
    }

    private void RegisterDefaultFusionPatterns()
    {
        // Element-wise chain pattern
        RegisterFusionPattern("ElementWiseChain", new FusionPatternDefinition
        {
            Name = "ElementWiseChain",
            OpTypeSequence = new[] { "Add", "Mul", "ReLU", "Sigmoid", "Tanh" },
            MatchStrategy = PatternMatchers.MatchElementWiseChain,
            Strategy = FusionStrategy.Merge,
            Priority = 10
        });

        // Conv + Activation pattern
        RegisterFusionPattern("ConvActivation", new FusionPatternDefinition
        {
            Name = "ConvActivation",
            OpTypeSequence = new[] { "Conv2D", "ReLU" },
            MatchStrategy = PatternMatchers.MatchConvActivation,
            Strategy = FusionStrategy.Merge,
            Priority = 20
        });

        // Conv + BatchNorm pattern
        RegisterFusionPattern("ConvBatchNorm", new FusionPatternDefinition
        {
            Name = "ConvBatchNorm",
            OpTypeSequence = new[] { "Conv2D", "BatchNorm" },
            MatchStrategy = PatternMatchers.MatchConvBatchNorm,
            Strategy = FusionStrategy.Fold,
            Priority = 25
        });

        // Linear + Activation pattern
        RegisterFusionPattern("LinearActivation", new FusionPatternDefinition
        {
            Name = "LinearActivation",
            OpTypeSequence = new[] { "Linear", "ReLU" },
            MatchStrategy = PatternMatchers.MatchLinearActivation,
            Strategy = FusionStrategy.Merge,
            Priority = 20
        });
    }
}
