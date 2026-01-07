namespace MlFramework.Inference.PagedAttention.Sharing;

/// <summary>
/// Information about a beam in beam search.
/// </summary>
public class BeamInfo
{
    public int BeamIndex { get; set; }
    public int BaseSequenceId { get; set; }
    public int DivergencePoint { get; set; }
}
