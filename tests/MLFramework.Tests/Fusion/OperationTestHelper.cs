using MLFramework.Core;
using MLFramework.Fusion;

namespace MLFramework.Tests.Fusion;

/// <summary>
/// Helper class to create test operations
/// </summary>
internal static class OperationTestHelper
{
    public static Operation CreateOperation(
        string type,
        DataType dataType = DataType.Float32,
        TensorLayout layout = TensorLayout.NCHW,
        TensorShape? inputShape = null,
        TensorShape? outputShape = null,
        Dictionary<string, object>? attributes = null)
    {
        var shape = inputShape ?? TensorShape.Create(1, 3, 224, 224);

        return new TestOperation
        {
            Id = Guid.NewGuid().ToString(),
            Type = type,
            Name = $"{type}_{Guid.NewGuid():N}",
            DataType = dataType,
            Layout = layout,
            InputShape = shape,
            OutputShape = outputShape ?? shape,
            Inputs = new[] { $"input_{Guid.NewGuid():N}" },
            Outputs = new[] { $"output_{Guid.NewGuid():N}" },
            Attributes = attributes ?? new Dictionary<string, object>()
        };
    }

    public static Operation CreateConvOp()
    {
        return CreateOperation("Conv2D");
    }

    public static Operation CreateReluOp()
    {
        return CreateOperation("ReLU");
    }

    public static Operation CreateAddOp()
    {
        return CreateOperation("Add");
    }

    public static Operation CreateLinearOp()
    {
        return CreateOperation("Linear");
    }

    public static Operation CreateBatchNormOp(bool training = false)
    {
        return CreateOperation("BatchNorm", attributes: new Dictionary<string, object>
        {
            { "training", training }
        });
    }
}

/// <summary>
/// Test implementation of Operation for unit tests
/// </summary>
internal record TestOperation(
    string Id,
    string Type,
    string Name,
    DataType DataType,
    TensorLayout Layout,
    TensorShape InputShape,
    TensorShape OutputShape,
    IReadOnlyList<string> Inputs,
    IReadOnlyList<string> Outputs,
    IReadOnlyDictionary<string, object> Attributes
) : Operation;
