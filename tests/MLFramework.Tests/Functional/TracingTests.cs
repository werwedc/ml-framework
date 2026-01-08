using Xunit;
using MLFramework.Functional.Tracing;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Functional
{
    public class TraceNodeTests
    {
        [Fact]
        public void Constructor_ShouldInitializeCorrectly()
        {
            // Arrange
            var shape = new TensorShape(3, 4);
            var inputs = new TraceNode[0];

            // Act
            var node = new TraceNode("add", inputs, shape, TensorType.Float32);

            // Assert
            Assert.Equal("add", node.OperationName);
            Assert.Empty(node.Inputs);
            Assert.Equal(shape, node.OutputShape);
            Assert.Equal(TensorType.Float32, node.OutputType);
            Assert.Empty(node.Attributes);
        }

        [Fact]
        public void Constructor_ShouldStoreAttributes()
        {
            // Arrange
            var shape = new TensorShape(3, 4);
            var attrs = new Dictionary<string, object>
            {
                { "axis", 0 },
                { "alpha", 1.0f }
            };

            // Act
            var node = new TraceNode("relu", Array.Empty<TraceNode>(), shape, TensorType.Float32, attrs);

            // Assert
            Assert.Equal(2, node.Attributes.Count);
            Assert.Equal(0, node.Attributes["axis"]);
            Assert.Equal(1.0f, node.Attributes["alpha"]);
        }

        [Fact]
        public void ToString_ShouldReturnCorrectFormat()
        {
            // Arrange
            var shape = new TensorShape(3, 4);
            var node = new TraceNode("matmul", Array.Empty<TraceNode>(), shape, TensorType.Float32);

            // Act
            var str = node.ToString();

            // Assert
            Assert.Contains("matmul", str);
            Assert.Contains("[3, 4]", str);
        }

        [Fact]
        public void Id_ShouldBeUnique()
        {
            // Arrange & Act
            var node1 = new TraceNode("op1", Array.Empty<TraceNode>(), new TensorShape(3), TensorType.Float32);
            var node2 = new TraceNode("op2", Array.Empty<TraceNode>(), new TensorShape(3), TensorType.Float32);

            // Assert
            Assert.NotEqual(node1.Id, node2.Id);
        }
    }

    public class TensorShapeTests
    {
        [Fact]
        public void Constructor_ShouldInitializeCorrectly()
        {
            // Act
            var shape = new TensorShape(2, 3, 4);

            // Assert
            Assert.Equal(3, shape.Rank);
            Assert.Equal(2, shape[0]);
            Assert.Equal(3, shape[1]);
            Assert.Equal(4, shape[2]);
        }

        [Fact]
        public void TotalElements_ShouldCalculateCorrectly()
        {
            // Act
            var shape = new TensorShape(2, 3, 4);

            // Assert
            Assert.Equal(24, shape.TotalElements);
        }

        [Fact]
        public void Scalar_ShouldHaveRankZero()
        {
            // Act
            var shape = TensorShape.Scalar;

            // Assert
            Assert.Equal(0, shape.Rank);
            Assert.Equal(1, shape.TotalElements);
        }

        [Fact]
        public void Equals_ShouldReturnTrueForSameShape()
        {
            // Arrange
            var shape1 = new TensorShape(2, 3);
            var shape2 = new TensorShape(2, 3);

            // Assert
            Assert.True(shape1.Equals(shape2));
        }

        [Fact]
        public void Equals_ShouldReturnFalseForDifferentShape()
        {
            // Arrange
            var shape1 = new TensorShape(2, 3);
            var shape2 = new TensorShape(3, 2);

            // Assert
            Assert.False(shape1.Equals(shape2));
        }

        [Fact]
        public void GetHashCode_ShouldBeSameForEqualShapes()
        {
            // Arrange
            var shape1 = new TensorShape(2, 3);
            var shape2 = new TensorShape(2, 3);

            // Assert
            Assert.Equal(shape1.GetHashCode(), shape2.GetHashCode());
        }

        [Fact]
        public void ToString_ShouldReturnCorrectFormat()
        {
            // Arrange
            var shape = new TensorShape(2, 3, 4);

            // Act
            var str = shape.ToString();

            // Assert
            Assert.Equal("[2, 3, 4]", str);
        }
    }

    public class TracedTensorTests
    {
        [Fact]
        public void Create_ShouldCreateTracedTensorWithNode()
        {
            // Arrange
            var tensor = new Tensor(new[] { 1f, 2f, 3f });

            // Act
            var traced = TracedTensor.Create(tensor, "input");

            // Assert
            Assert.NotNull(traced.Underlying);
            Assert.NotNull(traced.Node);
            Assert.Equal("input", traced.Node.OperationName);
        }

        [Fact]
        public void Add_ShouldCreateNewNode()
        {
            // Arrange
            var tensor1 = new Tensor(new[] { 1f, 2f });
            var tensor2 = new Tensor(new[] { 3f, 4f });
            var traced1 = TracedTensor.Create(tensor1, "input1");
            var traced2 = TracedTensor.Create(tensor2, "input2");

            // Act
            var result = traced1.Add(traced2);

            // Assert
            Assert.NotNull(result.Node);
            Assert.Equal("add", result.Node.OperationName);
            Assert.Equal(2, result.Node.Inputs.Length);
            Assert.Same(traced1.Node, result.Node.Inputs[0]);
            Assert.Same(traced2.Node, result.Node.Inputs[1]);
        }

        [Fact]
        public void Multiply_ShouldCreateNewNode()
        {
            // Arrange
            var tensor1 = new Tensor(new[] { 2f, 3f });
            var tensor2 = new Tensor(new[] { 4f, 5f });
            var traced1 = TracedTensor.Create(tensor1, "input1");
            var traced2 = TracedTensor.Create(tensor2, "input2");

            // Act
            var result = traced1.Multiply(traced2);

            // Assert
            Assert.NotNull(result.Node);
            Assert.Equal("multiply", result.Node.OperationName);
        }

        [Fact]
        public void ReLU_ShouldCreateNewNode()
        {
            // Arrange
            var tensor = new Tensor(new[] { -1f, 2f });
            var traced = TracedTensor.Create(tensor, "input");

            // Act
            var result = traced.ReLU();

            // Assert
            Assert.NotNull(result.Node);
            Assert.Equal("relu", result.Node.OperationName);
            Assert.Single(result.Node.Inputs);
            Assert.Same(traced.Node, result.Node.Inputs[0]);
        }

        [Fact]
        public void ImplicitConversion_ShouldReturnUnderlyingTensor()
        {
            // Arrange
            var tensor = new Tensor(new[] { 1f, 2f });
            var traced = TracedTensor.Create(tensor, "input");

            // Act
            Tensor result = traced;

            // Assert
            Assert.Same(tensor, result);
        }
    }

    public class TraceContextTests
    {
        [Fact]
        public void Constructor_ShouldSetAsCurrent()
        {
            // Act
            using (var trace = new TraceContext())
            {
                // Assert
                Assert.Same(trace, TraceContext.Current);
                Assert.True(trace.IsActive);
            }

            // After disposal, should not be active
            Assert.Null(TraceContext.Current);
        }

        [Fact]
        public void RecordNode_ShouldAddToNodes()
        {
            // Arrange
            using (var trace = new TraceContext())
            {
                var node = new TraceNode("op", Array.Empty<TraceNode>(),
                                       new TensorShape(3), TensorType.Float32);

                // Act
                trace.RecordNode(node);

                // Assert
                Assert.Single(trace.Nodes);
                Assert.Same(node, trace.Nodes[0]);
            }
        }

        [Fact]
        public void RecordNode_ShouldThrowWhenNotActive()
        {
            // Arrange
            var trace = new TraceContext();
            var node = new TraceNode("op", Array.Empty<TraceNode>(),
                                   new TensorShape(3), TensorType.Float32);

            // Act & Assert
            // Dispose first to make it inactive
            trace.Dispose();
            Assert.Throws<InvalidOperationException>(() => trace.RecordNode(node));
        }

        [Fact]
        public void RegisterOutput_ShouldStoreNamedOutput()
        {
            // Arrange
            using (var trace = new TraceContext())
            {
                var node = new TraceNode("output", Array.Empty<TraceNode>(),
                                       new TensorShape(3), TensorType.Float32);

                // Act
                trace.RegisterOutput("final", node);

                // Assert
                Assert.True(trace.NamedOutputs.ContainsKey("final"));
                Assert.Same(node, trace.NamedOutputs["final"]);
            }
        }

        [Fact]
        public void Dispose_ShouldClearCurrent()
        {
            // Arrange
            var trace = new TraceContext();

            // Act
            trace.Dispose();

            // Assert
            Assert.Null(TraceContext.Current);
            Assert.False(trace.IsActive);
        }

        [Fact]
        public void ToString_ShouldReturnFormattedTrace()
        {
            // Arrange
            using (var trace = new TraceContext())
            {
                var node1 = new TraceNode("input1", Array.Empty<TraceNode>(),
                                         new TensorShape(3), TensorType.Float32);
                var node2 = new TraceNode("input2", Array.Empty<TraceNode>(),
                                         new TensorShape(3), TensorType.Float32);
                var node3 = new TraceNode("add", new[] { node1, node2 },
                                         new TensorShape(3), TensorType.Float32);

                trace.RecordNode(node1);
                trace.RecordNode(node2);
                trace.RecordNode(node3);

                // Act
                var str = trace.ToString();

                // Assert
                Assert.Contains("Trace:", str);
                Assert.Contains("input1([3])", str);
                Assert.Contains("input2([3])", str);
                Assert.Contains("add([3])", str);
            }
        }
    }
}
