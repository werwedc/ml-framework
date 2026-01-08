using Xunit;
using MLFramework.Functional;
using RitterFramework.Core.Tensor;
using System;

namespace MLFramework.Tests.Functional
{
    /// <summary>
    /// Tests for the core functional transformation interfaces.
    /// </summary>
    public class CoreInterfacesTests
    {
        #region TransformationRegistry Tests

        public class TransformationRegistryTests
        {
            [Fact]
            public void Register_ShouldStoreTransformation()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;
                var transform = new DummyTransform();

                // Act
                TransformationRegistry.Register(func, transform);

                // Assert
                var transforms = TransformationRegistry.GetTransformations(func);
                Assert.Single(transforms);
                Assert.Same(transform, transforms[0]);
            }

            [Fact]
            public void Register_ShouldAllowMultipleTransformations()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;
                var transform1 = new DummyTransform();
                var transform2 = new DummyTransform();

                // Act
                TransformationRegistry.Register(func, transform1);
                TransformationRegistry.Register(func, transform2);

                // Assert
                var transforms = TransformationRegistry.GetTransformations(func);
                Assert.Equal(2, transforms.Count);
            }

            [Fact]
            public void GetTransformations_ShouldReturnEmptyForUnregistered()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;

                // Act
                var transforms = TransformationRegistry.GetTransformations(func);

                // Assert
                Assert.Empty(transforms);
            }

            [Fact]
            public void Clear_ShouldRemoveAllTransformations()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;
                TransformationRegistry.Register(func, new DummyTransform());

                // Act
                TransformationRegistry.Clear();

                // Assert
                var transforms = TransformationRegistry.GetTransformations(func);
                Assert.Empty(transforms);
            }

            [Fact]
            public void Register_ShouldThrowForNullDelegate()
            {
                // Act & Assert
                Assert.Throws<ArgumentNullException>(() =>
                    TransformationRegistry.Register(null, new DummyTransform()));
            }

            [Fact]
            public void Register_ShouldThrowForNullTransform()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;

                // Act & Assert
                Assert.Throws<ArgumentNullException>(() =>
                    TransformationRegistry.Register(func, null));
            }

            [Fact]
            public void Unregister_ShouldRemoveTransformations()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;
                TransformationRegistry.Register(func, new DummyTransform());

                // Act
                var result = TransformationRegistry.Unregister(func);

                // Assert
                Assert.True(result);
                Assert.Empty(TransformationRegistry.GetTransformations(func));
            }

            [Fact]
            public void Unregister_ShouldReturnFalseForUnregistered()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;

                // Act
                var result = TransformationRegistry.Unregister(func);

                // Assert
                Assert.False(result);
            }

            [Fact]
            public void HasTransformations_ShouldReturnCorrectValue()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;

                // Act & Assert
                Assert.False(TransformationRegistry.HasTransformations(func));

                TransformationRegistry.Register(func, new DummyTransform());
                Assert.True(TransformationRegistry.HasTransformations(func));
            }

            [Fact]
            public void GetTransformationsByType_ShouldFilterCorrectly()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;
                TransformationRegistry.Register(func, new DummyTransform());
                TransformationRegistry.Register(func, new DummyTransform2());

                // Act
                var transforms = TransformationRegistry.GetTransformationsByType(func, TransformationType.Composition);

                // Assert
                Assert.Single(transforms);
                Assert.Equal("dummy2", transforms[0].Name);
            }

            [Fact]
            public void HasTransformationOfType_ShouldReturnCorrectValue()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;
                TransformationRegistry.Register(func, new DummyTransform());

                // Act & Assert
                Assert.True(TransformationRegistry.HasTransformationOfType(func, TransformationType.Composition));
                Assert.False(TransformationRegistry.HasTransformationOfType(func, TransformationType.Vectorization));
            }

            [Fact]
            public void Count_ShouldReturnNumberOfRegisteredDelegates()
            {
                // Arrange
                Func<Tensor, Tensor> func1 = t => t;
                Func<Tensor, Tensor> func2 = t => t * 2;
                TransformationRegistry.Register(func1, new DummyTransform());
                TransformationRegistry.Register(func2, new DummyTransform());

                // Act
                var count = TransformationRegistry.Count();

                // Assert
                Assert.Equal(2, count);
            }

            [Fact]
            public void TotalTransformationCount_ShouldReturnTotalTransformations()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;
                TransformationRegistry.Register(func, new DummyTransform());
                TransformationRegistry.Register(func, new DummyTransform2());

                // Act
                var count = TransformationRegistry.TotalTransformationCount();

                // Assert
                Assert.Equal(2, count);
            }

            [Fact]
            public void GetTransformations_ShouldReturnCopyNotReference()
            {
                // Arrange
                Func<Tensor, Tensor> func = t => t;
                TransformationRegistry.Register(func, new DummyTransform());
                var transforms1 = TransformationRegistry.GetTransformations(func);

                // Act
                var transforms2 = TransformationRegistry.GetTransformations(func);
                transforms2.Clear();

                // Assert - Clearing the copy should not affect the registry
                var transforms3 = TransformationRegistry.GetTransformations(func);
                Assert.Single(transforms3);
            }
        }

        #endregion

        #region TransformationContext Tests

        public class TransformationContextTests
        {
            [Fact]
            public void Constructor_ShouldInitializeWithDefaults()
            {
                // Act
                var context = new TransformationContext();

                // Assert
                Assert.False(context.DebugMode);
                Assert.NotNull(context.Metadata);
                Assert.Empty(context.Metadata);
                Assert.Null(context.Parent);
            }

            [Fact]
            public void Constructor_WithParent_ShouldSetParent()
            {
                // Arrange
                var parent = new TransformationContext();

                // Act
                var child = new TransformationContext(parent);

                // Assert
                Assert.Same(parent, child.Parent);
            }

            [Fact]
            public void Constructor_WithDebugParent_ShouldInheritDebugMode()
            {
                // Arrange
                var parent = new TransformationContext { DebugMode = true };

                // Act
                var child = new TransformationContext(parent);

                // Assert
                Assert.True(child.DebugMode);
            }

            [Fact]
            public void Metadata_ShouldStoreAndRetrieveValues()
            {
                // Arrange
                var context = new TransformationContext();

                // Act
                context.Metadata["key"] = "value";

                // Assert
                Assert.Equal("value", context.Metadata["key"]);
            }

            [Fact]
            public void DebugMode_ShouldBeSettable()
            {
                // Arrange
                var context = new TransformationContext();

                // Act
                context.DebugMode = true;

                // Assert
                Assert.True(context.DebugMode);
            }

            [Fact]
            public void GetMetadata_ShouldReturnStoredValue()
            {
                // Arrange
                var context = new TransformationContext();
                context.Metadata["key"] = 42;

                // Act
                var value = context.GetMetadata<int>("key");

                // Assert
                Assert.Equal(42, value);
            }

            [Fact]
            public void GetMetadata_ShouldReturnDefaultForMissing()
            {
                // Arrange
                var context = new TransformationContext();

                // Act
                var value = context.GetMetadata<int>("missing");

                // Assert
                Assert.Equal(0, value);
            }

            [Fact]
            public void GetMetadata_ShouldReturnDefaultForTypeMismatch()
            {
                // Arrange
                var context = new TransformationContext();
                context.Metadata["key"] = "string";

                // Act
                var value = context.GetMetadata<int>("key");

                // Assert
                Assert.Equal(0, value);
            }

            [Fact]
            public void GetMetadata_ShouldSearchParent()
            {
                // Arrange
                var parent = new TransformationContext();
                parent.Metadata["key"] = 42;
                var child = new TransformationContext(parent);

                // Act
                var value = child.GetMetadata<int>("key");

                // Assert
                Assert.Equal(42, value);
            }

            [Fact]
            public void GetMetadata_WithSearchParentsFalse_ShouldNotSearchParent()
            {
                // Arrange
                var parent = new TransformationContext();
                parent.Metadata["key"] = 42;
                var child = new TransformationContext(parent);

                // Act
                var value = child.GetMetadata<int>("key", searchParents: false);

                // Assert
                Assert.Equal(0, value);
            }

            [Fact]
            public void SetMetadata_ShouldStoreValue()
            {
                // Arrange
                var context = new TransformationContext();

                // Act
                context.SetMetadata("key", 42);

                // Assert
                Assert.Equal(42, context.Metadata["key"]);
            }

            [Fact]
            public void CreateChildContext_ShouldSetParent()
            {
                // Arrange
                var parent = new TransformationContext();

                // Act
                var child = parent.CreateChildContext();

                // Assert
                Assert.Same(parent, child.Parent);
            }

            [Fact]
            public void CreateChildContext_ShouldInheritDebugMode()
            {
                // Arrange
                var parent = new TransformationContext { DebugMode = true };

                // Act
                var child = parent.CreateChildContext();

                // Assert
                Assert.True(child.DebugMode);
            }

            [Fact]
            public void CreateChildContext_ShouldNotShareMetadata()
            {
                // Arrange
                var parent = new TransformationContext();
                parent.Metadata["key"] = 42;

                // Act
                var child = parent.CreateChildContext();
                child.Metadata["key"] = 100;

                // Assert
                Assert.Equal(42, parent.Metadata["key"]);
                Assert.Equal(100, child.Metadata["key"]);
            }
        }

        #endregion

        #region BaseTransformation Tests

        public class BaseTransformationTests
        {
            [Fact]
            public void Constructor_ShouldSetProperties()
            {
                // Arrange
                var context = new TransformationContext();

                // Act
                var transform = new DummyTransform(context);

                // Assert
                Assert.Equal("dummy", transform.Name);
                Assert.Equal(TransformationType.Composition, transform.Type);
                Assert.Same(context, transform.Context);
            }

            [Fact]
            public void Constructor_ShouldCreateDefaultContext()
            {
                // Act
                var transform = new DummyTransform();

                // Assert
                Assert.NotNull(transform.Context);
                Assert.False(transform.Context.DebugMode);
            }

            [Fact]
            public void ValidateDelegate_ShouldThrowForNull()
            {
                // Arrange
                var transform = new DummyTransform();

                // Act & Assert
                Assert.Throws<ArgumentNullException>(() => transform.ValidateDelegate(null));
            }

            [Fact]
            public void ValidateDelegate_ShouldThrowForNonTensorFunction()
            {
                // Arrange
                var transform = new DummyTransform();
                Func<int, int> func = x => x;

                // Act & Assert
                Assert.Throws<ArgumentException>(() => transform.ValidateDelegate(func));
            }

            [Fact]
            public void ValidateDelegate_ShouldAcceptTensorFunction()
            {
                // Arrange
                var transform = new DummyTransform();
                Func<Tensor, Tensor> func = t => t;

                // Act & Assert
                var exception = Record.Exception(() => transform.ValidateDelegate(func));
                Assert.Null(exception);
            }

            [Fact]
            public void ValidateDelegate_ShouldAcceptMultipleTensorParameters()
            {
                // Arrange
                var transform = new DummyTransform();
                Func<Tensor, Tensor, Tensor> func = (a, b) => a + b;

                // Act & Assert
                var exception = Record.Exception(() => transform.ValidateDelegate(func));
                Assert.Null(exception);
            }

            [Fact]
            public void IsFunctionPure_ShouldReturnTrueForNoAttribute()
            {
                // Arrange
                var transform = new DummyTransform();
                Func<Tensor, Tensor> func = t => t;

                // Act
                var isPure = transform.IsFunctionPure(func);

                // Assert
                Assert.True(isPure);
            }

            [Fact]
            public void IsFunctionPure_ShouldReturnFalseForImpureAttribute()
            {
                // Arrange
                var transform = new DummyTransform();

                // Act
                var isPure = transform.IsFunctionPure(ImpureFunction);

                // Assert
                Assert.False(isPure);
            }

            [Fact]
            public void GetTensorFunctionAttribute_ShouldReturnAttribute()
            {
                // Arrange
                var transform = new DummyTransform();

                // Act
                var attr = transform.GetTensorFunctionAttribute(FunctionWithAttribute);

                // Assert
                Assert.NotNull(attr);
                Assert.Equal("[10]", attr.InputShapes[0]);
                Assert.Equal("[5]", attr.OutputShape);
            }

            [Fact]
            public void GetTensorFunctionAttribute_ShouldReturnNullForNoAttribute()
            {
                // Arrange
                var transform = new DummyTransform();
                Func<Tensor, Tensor> func = t => t;

                // Act
                var attr = transform.GetTensorFunctionAttribute(func);

                // Assert
                Assert.Null(attr);
            }

            private Tensor ImpureFunction(Tensor t) => t;

            [TensorFunction(new[] { "[10]" }, "[5]")]
            private Tensor FunctionWithAttribute(Tensor t) => t;
        }

        #endregion

        #region TensorFunctionAttribute Tests

        public class TensorFunctionAttributeTests
        {
            [Fact]
            public void Constructor_ShouldSetDefaults()
            {
                // Act
                var attr = new TensorFunctionAttribute();

                // Assert
                Assert.True(attr.IsPure);
                Assert.Null(attr.InputShapes);
                Assert.Null(attr.OutputShape);
                Assert.Null(attr.Description);
            }

            [Fact]
            public void Constructor_WithShapes_ShouldSetProperties()
            {
                // Arrange
                var inputShapes = new[] { "[10]", "[20]" };

                // Act
                var attr = new TensorFunctionAttribute(inputShapes, "[5]");

                // Assert
                Assert.Equal(inputShapes, attr.InputShapes);
                Assert.Equal("[5]", attr.OutputShape);
            }

            [Fact]
            public void Constructor_Full_ShouldSetAllProperties()
            {
                // Arrange
                var inputShapes = new[] { "[10]", "[20]" };

                // Act
                var attr = new TensorFunctionAttribute(false, inputShapes, "[5]");

                // Assert
                Assert.False(attr.IsPure);
                Assert.Equal(inputShapes, attr.InputShapes);
                Assert.Equal("[5]", attr.OutputShape);
            }

            [Fact]
            public void Properties_ShouldBeSettable()
            {
                // Arrange
                var attr = new TensorFunctionAttribute();

                // Act
                attr.IsPure = false;
                attr.InputShapes = new[] { "[10]", "[20]" };
                attr.OutputShape = "[5]";
                attr.Description = "Test function";

                // Assert
                Assert.False(attr.IsPure);
                Assert.Equal(new[] { "[10]", "[20]" }, attr.InputShapes);
                Assert.Equal("[5]", attr.OutputShape);
                Assert.Equal("Test function", attr.Description);
            }
        }

        #endregion

        #region TransformationType Enum Tests

        public class TransformationTypeTests
        {
            [Fact]
            public void ShouldHaveCorrectValues()
            {
                // Assert
                Assert.Equal(0, (int)TransformationType.Vectorization);
                Assert.Equal(1, (int)TransformationType.Parallelization);
                Assert.Equal(2, (int)TransformationType.Compilation);
                Assert.Equal(3, (int)TransformationType.Composition);
            }

            [Fact]
            public void ShouldHaveAllDefinedValues()
            {
                // Act & Assert - Verify we can cast all values
                var values = Enum.GetValues<TransformationType>();
                Assert.Equal(4, values.Length);
                Assert.Contains(TransformationType.Vectorization, values);
                Assert.Contains(TransformationType.Parallelization, values);
                Assert.Contains(TransformationType.Compilation, values);
                Assert.Contains(TransformationType.Composition, values);
            }
        }

        #endregion

        #region IFunctionalTransformation Interface Tests

        public class IFunctionalTransformationTests
        {
            [Fact]
            public void DummyTransform_ShouldImplementInterface()
            {
                // Arrange
                var transform = new DummyTransform();

                // Assert
                Assert.IsAssignableFrom<IFunctionalTransformation>(transform);
            }

            [Fact]
            public void Transform_ShouldReturnDelegate()
            {
                // Arrange
                var transform = new DummyTransform();
                Func<Tensor, Tensor> func = t => t;

                // Act
                var result = transform.Transform(func);

                // Assert
                Assert.NotNull(result);
                Assert.IsAssignableFrom<Func<Tensor, Tensor>>(result);
            }

            [Fact]
            public void Name_ShouldBeSet()
            {
                // Arrange
                var transform = new DummyTransform();

                // Assert
                Assert.Equal("dummy", transform.Name);
            }

            [Fact]
            public void Type_ShouldBeSet()
            {
                // Arrange
                var transform = new DummyTransform();

                // Assert
                Assert.Equal(TransformationType.Composition, transform.Type);
            }
        }

        #endregion

        #region Dummy Transform Classes for Testing

        /// <summary>
        /// Dummy transform for testing BaseTransformation
        /// </summary>
        internal class DummyTransform : BaseTransformation
        {
            public DummyTransform() : base("dummy", TransformationType.Composition) { }

            public DummyTransform(TransformationContext context) : base("dummy", TransformationType.Composition, context) { }

            public override Delegate Transform(Delegate original)
            {
                return original;
            }
        }

        /// <summary>
        /// Another dummy transform for testing
        /// </summary>
        internal class DummyTransform2 : BaseTransformation
        {
            public DummyTransform2() : base("dummy2", TransformationType.Vectorization) { }

            public override Delegate Transform(Delegate original)
            {
                return original;
            }
        }

        #endregion
    }
}
