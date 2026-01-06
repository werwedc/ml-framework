namespace MLFramework.Amp
{
    /// <summary>
    /// AutoCast mode for precision conversion
    /// </summary>
    public enum AutoCastMode
    {
        /// <summary>
        /// Cast to FP16 (Half precision)
        /// </summary>
        Fp16 = 0,

        /// <summary>
        /// Cast to BF16 (Brain Float)
        /// </summary>
        Bf16 = 1,

        /// <summary>
        /// No casting (keep original precision)
        /// </summary>
        None = 2
    }
}
