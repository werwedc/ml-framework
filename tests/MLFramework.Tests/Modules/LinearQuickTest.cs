using RitterFramework.Core.Tensor;
using MLFramework.Modules;

namespace MLFramework.Tests.Modules
{
    /// <summary>
    /// Quick verification that Linear module works
    /// </summary>
    public class LinearQuickTest
    {
        public static void RunQuickTest()
        {
            Console.WriteLine("Testing Linear module...");

            var linear = new Linear(inFeatures: 64, outFeatures: 128);
            var input = new Tensor(new float[10 * 64], new[] { 10, 64 });

            var output = linear.Forward(input);

            Console.WriteLine($"Linear test passed!");
            Console.WriteLine($"Input shape: {string.Join(", ", input.Shape)}");
            Console.WriteLine($"Output shape: {string.Join(", ", output.Shape)}");
            Console.WriteLine($"InFeatures: {linear.InFeatures}");
            Console.WriteLine($"OutFeatures: {linear.OutFeatures}");
        }
    }
}
