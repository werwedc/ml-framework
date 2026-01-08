using MLFramework.Shapes;
using MLFramework.Core;

namespace MLFramework.Fusion.Dynamic;

/// <summary>
/// Represents a node in a fusion graph that contains multiple fusible operations
/// </summary>
public class FusionNode
{
    private readonly List<Operation> _operations = new();
    private readonly List<SymbolicShape> _inputShapes = new();
    private readonly List<SymbolicShape> _outputShapes = new();

    /// <summary>
    /// Gets the operations in this fusion node
    /// </summary>
    public IReadOnlyList<Operation> Operations => _operations.AsReadOnly();

    /// <summary>
    /// Gets the input shapes for this fusion node
    /// </summary>
    public IReadOnlyList<SymbolicShape> InputShapes => _inputShapes.AsReadOnly();

    /// <summary>
    /// Gets the output shapes for this fusion node
    /// </summary>
    public IReadOnlyList<SymbolicShape> OutputShapes => _outputShapes.AsReadOnly();

    /// <summary>
    /// Gets the unique identifier for this fusion node
    /// </summary>
    public string FusionId { get; init; } = Guid.NewGuid().ToString("N");

    /// <summary>
    /// Initializes a new instance of the FusionNode class
    /// </summary>
    public FusionNode()
    {
        FusionId = Guid.NewGuid().ToString("N")!;
    }

    /// <summary>
    /// Initializes a new instance of the FusionNode class with a specific ID
    /// </summary>
    /// <param name="fusionId">The fusion ID</param>
    public FusionNode(string fusionId)
    {
        FusionId = fusionId ?? throw new ArgumentNullException(nameof(fusionId));
    }

    /// <summary>
    /// Adds an operation to this fusion node
    /// </summary>
    /// <param name="op">The operation to add</param>
    public void AddOperation(Operation op)
    {
        if (op == null)
            throw new ArgumentNullException(nameof(op));

        _operations.Add(op);
    }

    /// <summary>
    /// Adds an input shape to this fusion node
    /// </summary>
    /// <param name="shape">The input shape to add</param>
    public void AddInputShape(SymbolicShape shape)
    {
        if (shape == null)
            throw new ArgumentNullException(nameof(shape));

        _inputShapes.Add(shape);
    }

    /// <summary>
    /// Adds an output shape to this fusion node
    /// </summary>
    /// <param name="shape">The output shape to add</param>
    public void AddOutputShape(SymbolicShape shape)
    {
        if (shape == null)
            throw new ArgumentNullException(nameof(shape));

        _outputShapes.Add(shape);
    }

    /// <summary>
    /// Determines whether this fusion node can be fused with the next operation
    /// </summary>
    /// <param name="nextOp">The next operation to potentially fuse</param>
    /// <param name="intermediateShapes">The intermediate shapes between operations</param>
    /// <returns>True if fusion is possible; otherwise, false</returns>
    public bool CanFuseWith(Operation nextOp, List<SymbolicShape> intermediateShapes)
    {
        if (nextOp == null)
            return false;

        if (intermediateShapes == null || intermediateShapes.Count == 0)
            return false;

        // Check if the output shapes of this node match the input shapes of the next operation
        if (_outputShapes.Count != intermediateShapes.Count)
            return false;

        for (int i = 0; i < _outputShapes.Count; i++)
        {
            if (!_outputShapes[i].Equals(intermediateShapes[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Gets a unique signature for this fusion node based on its operations and shapes
    /// </summary>
    /// <returns>A signature string</returns>
    public string GetFusedSignature()
    {
        var opTypes = string.Join("->", _operations.Select(op => op.Type));
        var inputSig = string.Join("|", _inputShapes.Select(s => s.ToString()));
        var outputSig = string.Join("|", _outputShapes.Select(s => s.ToString()));

        return $"{opTypes}:[{inputSig}]->[{outputSig}]";
    }

    /// <summary>
    /// Validates that this fusion node is well-formed
    /// </summary>
    /// <returns>True if the fusion node is valid; otherwise, false</returns>
    public bool ValidateFusion()
    {
        // Must have at least one operation
        if (_operations.Count == 0)
            return false;

        // Must have at least one input and one output shape
        if (_inputShapes.Count == 0 || _outputShapes.Count == 0)
            return false;

        // Validate that all operations have IDs
        if (_operations.Any(op => string.IsNullOrEmpty(op.Id)))
            return false;

        // Validate that all shapes are non-null
        if (_inputShapes.Any(s => s == null) || _outputShapes.Any(s => s == null))
            return false;

        return true;
    }

    /// <summary>
    /// Creates a clone of this fusion node
    /// </summary>
    /// <returns>A new FusionNode with cloned operations and shapes</returns>
    public FusionNode Clone()
    {
        var cloned = new FusionNode(Guid.NewGuid().ToString("N"));

        foreach (var op in _operations)
        {
            cloned.AddOperation(op);
        }

        foreach (var shape in _inputShapes)
        {
            cloned.AddInputShape(shape.Clone());
        }

        foreach (var shape in _outputShapes)
        {
            cloned.AddOutputShape(shape.Clone());
        }

        return cloned;
    }
}
