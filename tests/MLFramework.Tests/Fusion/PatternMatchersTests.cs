using NUnit.Framework;
using MLFramework.Core;
using MLFramework.Fusion;

namespace MLFramework.Tests.Fusion;

/// <summary>
/// Tests for PatternMatchers
/// </summary>
[TestFixture]
public class PatternMatchersTests
{
    [Test]
    public void MatchElementWiseChain_SingleOp_ReturnsFalse()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateAddOp()
        };

        var result = PatternMatchers.MatchElementWiseChain(ops);

        Assert.That(result, Is.False);
    }

    [Test]
    public void MatchElementWiseChain_MultipleElementWiseOps_ReturnsTrue()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateAddOp(),
            OperationTestHelper.CreateOperation("Mul"),
            OperationTestHelper.CreateReluOp()
        };

        var result = PatternMatchers.MatchElementWiseChain(ops);

        Assert.That(result, Is.True);
    }

    [Test]
    public void MatchElementWiseChain_IncompatibleShapes_ReturnsFalse()
    {
        var op1 = OperationTestHelper.CreateOperation("Add", outputShape: TensorShape.Create(1, 3, 224, 224));
        var op2 = OperationTestHelper.CreateOperation("Mul", inputShape: TensorShape.Create(1, 3, 112, 112));

        var result = PatternMatchers.MatchElementWiseChain(new[] { op1, op2 });

        Assert.That(result, Is.False);
    }

    [Test]
    public void MatchElementWiseChain_ContainsNonElementWise_ReturnsFalse()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateAddOp(),
            OperationTestHelper.CreateConvOp(),
            OperationTestHelper.CreateReluOp()
        };

        var result = PatternMatchers.MatchElementWiseChain(ops);

        Assert.That(result, Is.False);
    }

    [Test]
    public void MatchConvActivation_Conv2DReLU_ReturnsTrue()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateConvOp(),
            OperationTestHelper.CreateReluOp()
        };

        var result = PatternMatchers.MatchConvActivation(ops);

        Assert.That(result, Is.True);
    }

    [Test]
    public void MatchConvActivation_SingleOp_ReturnsFalse()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateConvOp()
        };

        var result = PatternMatchers.MatchConvActivation(ops);

        Assert.That(result, Is.False);
    }

    [Test]
    public void MatchConvActivation_WrongOrder_ReturnsFalse()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateReluOp(),
            OperationTestHelper.CreateConvOp()
        };

        var result = PatternMatchers.MatchConvActivation(ops);

        Assert.That(result, Is.False);
    }

    [Test]
    public void MatchConvBatchNorm_Conv2DBatchNorm_ReturnsTrue()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateConvOp(),
            OperationTestHelper.CreateBatchNormOp(training: false)
        };

        var result = PatternMatchers.MatchConvBatchNorm(ops);

        Assert.That(result, Is.True);
    }

    [Test]
    public void MatchConvBatchNorm_TrainingMode_ReturnsFalse()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateConvOp(),
            OperationTestHelper.CreateBatchNormOp(training: true)
        };

        var result = PatternMatchers.MatchConvBatchNorm(ops);

        Assert.That(result, Is.False);
    }

    [Test]
    public void MatchLinearActivation_LinearReLU_ReturnsTrue()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateLinearOp(),
            OperationTestHelper.CreateReluOp()
        };

        var result = PatternMatchers.MatchLinearActivation(ops);

        Assert.That(result, Is.True);
    }

    [Test]
    public void MatchLinearActivation_SingleOp_ReturnsFalse()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateLinearOp()
        };

        var result = PatternMatchers.MatchLinearActivation(ops);

        Assert.That(result, Is.False);
    }

    [Test]
    public void MatchConvActivation_Conv2DSigmoid_ReturnsTrue()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateConvOp(),
            OperationTestHelper.CreateOperation("Sigmoid")
        };

        var result = PatternMatchers.MatchConvActivation(ops);

        Assert.That(result, Is.True);
    }

    [Test]
    public void MatchConvActivation_Conv2DNonActivation_ReturnsFalse()
    {
        var ops = new[]
        {
            OperationTestHelper.CreateConvOp(),
            OperationTestHelper.CreateAddOp()
        };

        var result = PatternMatchers.MatchConvActivation(ops);

        Assert.That(result, Is.False);
    }
}
