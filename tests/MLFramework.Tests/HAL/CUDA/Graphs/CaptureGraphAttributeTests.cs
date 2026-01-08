using Microsoft.VisualStudio.TestTools.UnitTesting;
using MLFramework.HAL.CUDA.Graphs;
using MLFramework.HAL.CUDA.Graphs.Attributes;
using MLFramework.HAL.CUDA.Graphs.Proxies;

namespace MLFramework.Tests.HAL.CUDA.Graphs;

/// <summary>
/// Unit tests for CaptureGraphAttribute
/// </summary>
[TestClass]
public class CaptureGraphAttributeTests
{
    [TestMethod]
    public void CaptureGraphAttribute_DefaultConstructor_CreatesWithDefaults()
    {
        // Arrange & Act
        var attr = new CaptureGraphAttribute();

        // Assert
        Assert.IsNotNull(attr);
        Assert.AreEqual(string.Empty, attr.GraphName);
        Assert.AreEqual(3, attr.WarmupIterations);
        Assert.IsTrue(attr.EnableWeightUpdates);
        Assert.IsTrue(attr.EnableFallback);
    }

    [TestMethod]
    public void CaptureGraphAttribute_NameConstructor_CreatesWithSpecifiedName()
    {
        // Arrange & Act
        var attr = new CaptureGraphAttribute("TestGraph");

        // Assert
        Assert.IsNotNull(attr);
        Assert.AreEqual("TestGraph", attr.GraphName);
        Assert.AreEqual(3, attr.WarmupIterations);
        Assert.IsTrue(attr.EnableWeightUpdates);
        Assert.IsTrue(attr.EnableFallback);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void CaptureGraphAttribute_NameConstructor_NullName_ThrowsException()
    {
        // Act
        var attr = new CaptureGraphAttribute(null!);
    }

    [TestMethod]
    public void CaptureGraphAttribute_CanSetProperties()
    {
        // Arrange
        var attr = new CaptureGraphAttribute();

        // Act
        attr.GraphName = "CustomGraph";
        attr.WarmupIterations = 5;
        attr.EnableWeightUpdates = false;
        attr.EnableFallback = false;

        // Assert
        Assert.AreEqual("CustomGraph", attr.GraphName);
        Assert.AreEqual(5, attr.WarmupIterations);
        Assert.IsFalse(attr.EnableWeightUpdates);
        Assert.IsFalse(attr.EnableFallback);
    }
}
