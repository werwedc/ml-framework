using NUnit.Framework;

namespace MLFramework.Tests.Inference.ContinuousBatching;

[TestFixture]
public class RequestIdTests
{
    [Test]
    public void New_CreatesUniqueId()
    {
        // Arrange & Act
        var id1 = RequestId.New();
        var id2 = RequestId.New();

        // Assert
        Assert.That(id1, Is.Not.EqualTo(id2));
    }

    [Test]
    public void Empty_ReturnsEmptyGuid()
    {
        // Arrange & Act
        var emptyId = RequestId.Empty;

        // Assert
        Assert.That(emptyId.Id, Is.EqualTo(Guid.Empty));
    }

    [Test]
    public void Equality_SameGuid_AreEqual()
    {
        // Arrange
        var guid = Guid.NewGuid();
        var id1 = new RequestId(guid);
        var id2 = new RequestId(guid);

        // Act & Assert
        Assert.That(id1, Is.EqualTo(id2));
    }

    [Test]
    public void Equality_DifferentGuid_AreNotEqual()
    {
        // Arrange
        var id1 = new RequestId(Guid.NewGuid());
        var id2 = new RequestId(Guid.NewGuid());

        // Act & Assert
        Assert.That(id1, Is.Not.EqualTo(id2));
    }

    [Test]
    public void Struct_IsLightweight()
    {
        // Arrange & Act
        var id = RequestId.New();
        var size = System.Runtime.InteropServices.Marshal.SizeOf<RequestId>();

        // Assert
        // RequestId is a struct wrapping a Guid (16 bytes)
        Assert.That(size, Is.LessThanOrEqualTo(32)); // Should be 16 bytes
    }
}
