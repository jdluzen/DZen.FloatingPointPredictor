using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;

namespace DZen.FloatingPointPredictor
{
    // ── Floating-Point Predictor (TIFF Technical Note 3, Predictor=3) ──────────

    /// <summary>
    /// Implements the floating-point predictor defined in TIFF Technical Note 3
    /// (also known as TIFF Predictor = 3).  This is the same algorithm used by
    /// GDAL, libtiff, and the COG/ZSTD pipeline for float32 raster data.
    ///
    /// Algorithm overview
    /// ──────────────────
    /// Each float32 pixel is treated as 4 raw bytes.  Within a tile row of
    /// <c>width</c> floats the bytes are first reorganised into 4 byte-planes
    /// (byte-shuffle):
    ///
    ///   plane 0 ← byte 3 (MSB) of every float
    ///   plane 1 ← byte 2
    ///   plane 2 ← byte 1
    ///   plane 3 ← byte 0 (LSB) of every float
    ///
    /// Then each plane is horizontal-delta encoded: every byte is replaced by
    /// (current − previous).  This concentrates entropy, making subsequent
    /// entropy coders (ZSTD, LZW, Deflate) significantly more effective on
    /// floating-point raster data.
    ///
    /// Decoding reverses both steps: undo delta, then un-shuffle bytes back
    /// to interleaved float32 layout.
    ///
    /// Hardware dispatch
    /// ─────────────────
    /// The byte-shuffle step is the inner-loop bottleneck and is accelerated
    /// with SIMD intrinsics.  At runtime the fastest available path is chosen:
    ///
    ///   • AVX-512BW  (16 floats = 64 bytes per iteration; full 512-bit vpermb)
    ///   • AVX2       (8 floats = 32 bytes per iteration; vpshufb × 2 lanes)
    ///   • SSSE3/SSE  (4 floats = 16 bytes per iteration; _mm_shuffle_epi8)
    ///   • ARM NEON   (4 floats = 16 bytes per iteration; AdvSimd)
    ///   • Scalar     (portable, no SIMD)
    ///
    /// Thread safety: all public methods are stateless and thread-safe.
    /// </summary>
    public static class Fp32Predictor
    {
        // ── Public API ─────────────────────────────────────────────────────────

        /// <summary>
        /// Applies the floating-point predictor encode transform in-place to a
        /// single tile (or strip) stored in <paramref name="tile"/>.
        ///
        /// The tile is treated as <paramref name="rows"/> × <paramref name="width"/>
        /// float32 values in row-major order (total bytes = rows × width × 4).
        /// Each row is processed independently (as per the TIFF TN3 spec).
        /// </summary>
        /// <param name="tile">Raw tile bytes (float32, row-major).</param>
        /// <param name="width">Number of float32 pixels per row.</param>
        /// <param name="rows">Number of rows in the tile.</param>
        public static void Encode(Span<byte> tile, int width, int rows)
        {
            int rowBytes = width * 4;
            for (int r = 0; r < rows; r++)
                EncodeRow(tile.Slice(r * rowBytes, rowBytes), width);
        }

        /// <summary>
        /// Reverses the floating-point predictor transform in-place.
        /// Operates on each row independently; each row must be exactly
        /// <paramref name="width"/> × 4 bytes.
        /// </summary>
        public static void Decode(Span<byte> tile, int width, int rows)
        {
            int rowBytes = width * 4;
            for (int r = 0; r < rows; r++)
                DecodeRow(tile.Slice(r * rowBytes, rowBytes), width);
        }

        // ── Per-row encode ─────────────────────────────────────────────────────

        private static void EncodeRow(Span<byte> row, int width)
        {
            ShuffleForward(row, width);

            int stride = width;
            for (int plane = 0; plane < 4; plane++)
            {
                int off = plane * stride;
                for (int i = width - 1; i >= 1; i--)
                    row[off + i] -= row[off + i - 1];
            }
        }

        // ── Per-row decode ─────────────────────────────────────────────────────

        private static void DecodeRow(Span<byte> row, int width)
        {
            int stride = width;
            for (int plane = 0; plane < 4; plane++)
            {
                int off = plane * stride;
                for (int i = 1; i < width; i++)
                    row[off + i] += row[off + i - 1];
            }

            ShuffleInverse(row, width);
        }

        // ── Byte-shuffle kernel dispatch ───────────────────────────────────────

        // Forward shuffle: interleaved float32 → 4-plane byte layout
        //   input:  [B3 B2 B1 B0 | B3 B2 B1 B0 | ...]   (N floats)
        //   output: [B3 B3 … B3 | B2 B2 … B2 | B1 B1 … | B0 B0 …]
        //
        // Byte index mapping for float i (0-based):
        //   output[        i] = input[4i+3]   plane 0 (MSB)
        //   output[  width+i] = input[4i+2]   plane 1
        //   output[2*width+i] = input[4i+1]   plane 2
        //   output[3*width+i] = input[4i+0]   plane 3 (LSB)

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ShuffleForward(Span<byte> row, int width)
        {
            if (Avx512Vbmi.IsSupported && width >= 16)
                ShuffleForwardAvx512(row, width);
            else if (Avx2.IsSupported && width >= 8)
                ShuffleForwardAvx2(row, width);
            else if (Ssse3.IsSupported && width >= 4)
                ShuffleForwardSsse3(row, width);
            else if (AdvSimd.Arm64.IsSupported && width >= 4)
                ShuffleForwardNeon(row, width);
            else
                ShuffleForwardScalar(row, width);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ShuffleInverse(Span<byte> row, int width)
        {
            if (Avx512Vbmi.IsSupported && width >= 16)
                ShuffleInverseAvx512(row, width);
            else if (Avx2.IsSupported && width >= 8)
                ShuffleInverseAvx2(row, width);
            else if (Ssse3.IsSupported && width >= 4)
                ShuffleInverseSsse3(row, width);
            else if (AdvSimd.Arm64.IsSupported && width >= 4)
                ShuffleInverseNeon(row, width);
            else
                ShuffleInverseScalar(row, width);
        }

        // ── Scalar fallback ────────────────────────────────────────────────────

        private static void ShuffleForwardScalar(Span<byte> row, int width)
        {
            var tmp = new byte[width * 4];
            for (int i = 0; i < width; i++)
            {
                tmp[            i] = row[4 * i + 3];
                tmp[    width + i] = row[4 * i + 2];
                tmp[2 * width + i] = row[4 * i + 1];
                tmp[3 * width + i] = row[4 * i + 0];
            }
            tmp.CopyTo(row);
        }

        private static void ShuffleInverseScalar(Span<byte> row, int width)
        {
            var tmp = new byte[width * 4];
            for (int i = 0; i < width; i++)
            {
                tmp[4 * i + 3] = row[            i];
                tmp[4 * i + 2] = row[    width + i];
                tmp[4 * i + 1] = row[2 * width + i];
                tmp[4 * i + 0] = row[3 * width + i];
            }
            tmp.CopyTo(row);
        }

        // ── SSSE3 / SSE2 (4 floats = 16 bytes per iteration) ──────────────────

        // Shuffle control: extract every 4th byte starting at position p.
        // For 4 floats packed as [f0b3 f0b2 f0b1 f0b0 | f1b3 f1b2 f1b1 f1b0 | …]:
        //   plane0 (MSB): bytes 3,7,11,15 → shuf [3,7,11,15, 0x80,0x80,…]
        //   plane1:       bytes 2,6,10,14
        //   plane2:       bytes 1,5,9,13
        //   plane3 (LSB): bytes 0,4,8,12

        private static void ShuffleForwardSsse3(Span<byte> row, int width)
        {
            if (!Ssse3.IsSupported) { ShuffleForwardScalar(row, width); return; }

            var tmp = new byte[width * 4];
            int i = 0;

            var shuf0 = Vector128.Create((byte)3,  7, 11, 15, 0x80, 0x80, 0x80, 0x80,
                                                   0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
            var shuf1 = Vector128.Create((byte)2,  6, 10, 14, 0x80, 0x80, 0x80, 0x80,
                                                   0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
            var shuf2 = Vector128.Create((byte)1,  5,  9, 13, 0x80, 0x80, 0x80, 0x80,
                                                   0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
            var shuf3 = Vector128.Create((byte)0,  4,  8, 12, 0x80, 0x80, 0x80, 0x80,
                                                   0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

            ref byte srcRef = ref MemoryMarshal.GetReference(row);
            ref byte dstRef = ref MemoryMarshal.GetArrayDataReference(tmp);

            for (; i <= width - 4; i += 4)
            {
                ReadOnlySpan<byte> data = [1, 2, 3, 4];
                ref readonly byte data0 = ref MemoryMarshal.GetReference(data);
                Vector128<byte> vector2 = Vector128.LoadUnsafe(in data0);

                var src = Vector128.LoadUnsafe(ref Unsafe.Add(ref srcRef, i * 4));
                uint p0 = Ssse3.Shuffle(src, shuf0).AsUInt32().GetElement(0);
                uint p1 = Ssse3.Shuffle(src, shuf1).AsUInt32().GetElement(0);
                uint p2 = Ssse3.Shuffle(src, shuf2).AsUInt32().GetElement(0);
                uint p3 = Ssse3.Shuffle(src, shuf3).AsUInt32().GetElement(0);

                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef,             i), p0);
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef,     width + i), p1);
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef, 2 * width + i), p2);
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef, 3 * width + i), p3);
            }

            for (; i < width; i++)
            {
                tmp[            i] = row[4 * i + 3];
                tmp[    width + i] = row[4 * i + 2];
                tmp[2 * width + i] = row[4 * i + 1];
                tmp[3 * width + i] = row[4 * i + 0];
            }
            tmp.CopyTo(row);
        }

        private static void ShuffleInverseSsse3(Span<byte> row, int width)
        {
            if (!Ssse3.IsSupported) { ShuffleInverseScalar(row, width); return; }

            var tmp = new byte[width * 4];
            int i = 0;

            ref byte srcRef = ref MemoryMarshal.GetReference(row);
            ref byte dstRef = ref MemoryMarshal.GetArrayDataReference(tmp);

            for (; i <= width - 4; i += 4)
            {
                uint b3 = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef,             i));
                uint b2 = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef,     width + i));
                uint b1 = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef, 2 * width + i));
                uint b0 = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef, 3 * width + i));

                var vb3 = Vector128.CreateScalar(b3).AsByte();
                var vb2 = Vector128.CreateScalar(b2).AsByte();
                var vb1 = Vector128.CreateScalar(b1).AsByte();
                var vb0 = Vector128.CreateScalar(b0).AsByte();

                var lo01 = Sse2.UnpackLow(vb0, vb1);
                var lo23 = Sse2.UnpackLow(vb2, vb3);
                var result = Sse2.UnpackLow(lo01.AsUInt16(), lo23.AsUInt16()).AsByte();

                result.StoreUnsafe(ref Unsafe.Add(ref dstRef, i * 4));
            }

            for (; i < width; i++)
            {
                tmp[4 * i + 3] = row[            i];
                tmp[4 * i + 2] = row[    width + i];
                tmp[4 * i + 1] = row[2 * width + i];
                tmp[4 * i + 0] = row[3 * width + i];
            }
            tmp.CopyTo(row);
        }

        // ── AVX2 (8 floats = 32 bytes per iteration) ───────────────────────────

        // AVX2 _mm256_shuffle_epi8 operates on two independent 128-bit lanes.
        // We load 32 bytes (8 floats) and extract 8 bytes per plane across both lanes.
        // Within each 128-bit lane the layout is identical to the SSSE3 case.

        private static void ShuffleForwardAvx2(Span<byte> row, int width)
        {
            if (!Avx2.IsSupported) { ShuffleForwardSsse3(row, width); return; }

            var tmp = new byte[width * 4];
            int i = 0;

            var shuf0 = Vector256.Create(
                (byte)3,  7, 11, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                      3,  7, 11, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
            var shuf1 = Vector256.Create(
                (byte)2,  6, 10, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                      2,  6, 10, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
            var shuf2 = Vector256.Create(
                (byte)1,  5,  9, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                      1,  5,  9, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
            var shuf3 = Vector256.Create(
                (byte)0,  4,  8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                      0,  4,  8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

            ref byte srcRef = ref MemoryMarshal.GetReference(row);
            ref byte dstRef = ref MemoryMarshal.GetArrayDataReference(tmp);

            for (; i <= width - 8; i += 8)
            {
                var src = Vector256.LoadUnsafe(ref Unsafe.Add(ref srcRef, i * 4));

                var v0 = Avx2.Shuffle(src, shuf0);
                var v1 = Avx2.Shuffle(src, shuf1);
                var v2 = Avx2.Shuffle(src, shuf2);
                var v3 = Avx2.Shuffle(src, shuf3);

                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef,             i),     v0.GetLower().AsUInt32().GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef,             i + 4), v0.GetUpper().AsUInt32().GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef,     width + i),     v1.GetLower().AsUInt32().GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef,     width + i + 4), v1.GetUpper().AsUInt32().GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef, 2 * width + i),     v2.GetLower().AsUInt32().GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef, 2 * width + i + 4), v2.GetUpper().AsUInt32().GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef, 3 * width + i),     v3.GetLower().AsUInt32().GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef, 3 * width + i + 4), v3.GetUpper().AsUInt32().GetElement(0));
            }

            for (; i < width; i++)
            {
                tmp[            i] = row[4 * i + 3];
                tmp[    width + i] = row[4 * i + 2];
                tmp[2 * width + i] = row[4 * i + 1];
                tmp[3 * width + i] = row[4 * i + 0];
            }
            tmp.CopyTo(row);
        }

        private static void ShuffleInverseAvx2(Span<byte> row, int width)
        {
            if (!Avx2.IsSupported) { ShuffleInverseSsse3(row, width); return; }

            var tmp = new byte[width * 4];
            int i = 0;

            ref byte srcRef = ref MemoryMarshal.GetReference(row);
            ref byte dstRef = ref MemoryMarshal.GetArrayDataReference(tmp);

            for (; i <= width - 8; i += 8)
            {
                uint b3_lo = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef,             i));
                uint b3_hi = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef,             i + 4));
                uint b2_lo = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef,     width + i));
                uint b2_hi = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef,     width + i + 4));
                uint b1_lo = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef, 2 * width + i));
                uint b1_hi = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef, 2 * width + i + 4));
                uint b0_lo = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef, 3 * width + i));
                uint b0_hi = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef, 3 * width + i + 4));

                var vb3l = Vector128.CreateScalar(b3_lo).AsByte();
                var vb2l = Vector128.CreateScalar(b2_lo).AsByte();
                var vb1l = Vector128.CreateScalar(b1_lo).AsByte();
                var vb0l = Vector128.CreateScalar(b0_lo).AsByte();
                var lo01l = Sse2.UnpackLow(vb0l, vb1l);
                var lo23l = Sse2.UnpackLow(vb2l, vb3l);
                var resLo = Sse2.UnpackLow(lo01l.AsUInt16(), lo23l.AsUInt16()).AsByte();

                var vb3h = Vector128.CreateScalar(b3_hi).AsByte();
                var vb2h = Vector128.CreateScalar(b2_hi).AsByte();
                var vb1h = Vector128.CreateScalar(b1_hi).AsByte();
                var vb0h = Vector128.CreateScalar(b0_hi).AsByte();
                var lo01h = Sse2.UnpackLow(vb0h, vb1h);
                var lo23h = Sse2.UnpackLow(vb2h, vb3h);
                var resHi = Sse2.UnpackLow(lo01h.AsUInt16(), lo23h.AsUInt16()).AsByte();

                resLo.StoreUnsafe(ref Unsafe.Add(ref dstRef, i * 4));
                resHi.StoreUnsafe(ref Unsafe.Add(ref dstRef, i * 4 + 16));
            }

            for (; i < width; i++)
            {
                tmp[4 * i + 3] = row[            i];
                tmp[4 * i + 2] = row[    width + i];
                tmp[4 * i + 1] = row[2 * width + i];
                tmp[4 * i + 0] = row[3 * width + i];
            }
            tmp.CopyTo(row);
        }

        // ── AVX-512BW (16 floats = 64 bytes per iteration) ─────────────────────

        // AVX-512BW vpermb: full 64-byte permute with arbitrary byte indices —
        // no lane restriction.  A single instruction shuffles all 16 floats at once.

        private static void ShuffleForwardAvx512(Span<byte> row, int width)
        {
            if (!Avx512Vbmi.IsSupported) { ShuffleForwardAvx2(row, width); return; }

            var tmp = new byte[width * 4];
            int i = 0;

            var perm0 = Vector512.Create(
                (byte) 3, 7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,
                       2, 6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,
                       1, 5, 9,13,17,21,25,29,33,37,41,45,49,53,57,61,
                       0, 4, 8,12,16,20,24,28,32,36,40,44,48,52,56,60);

            ref byte srcRef = ref MemoryMarshal.GetReference(row);
            ref byte dstRef = ref MemoryMarshal.GetArrayDataReference(tmp);

            for (; i <= width - 16; i += 16)
            {
                var src      = Vector512.LoadUnsafe(ref Unsafe.Add(ref srcRef, i * 4));
                var shuffled = Avx512Vbmi.PermuteVar64x8(src, perm0);
                shuffled.GetLower().GetLower().StoreUnsafe(ref Unsafe.Add(ref dstRef,             i));
                shuffled.GetLower().GetUpper().StoreUnsafe(ref Unsafe.Add(ref dstRef,     width + i));
                shuffled.GetUpper().GetLower().StoreUnsafe(ref Unsafe.Add(ref dstRef, 2 * width + i));
                shuffled.GetUpper().GetUpper().StoreUnsafe(ref Unsafe.Add(ref dstRef, 3 * width + i));
            }

            for (; i < width; i++)
            {
                tmp[            i] = row[4 * i + 3];
                tmp[    width + i] = row[4 * i + 2];
                tmp[2 * width + i] = row[4 * i + 1];
                tmp[3 * width + i] = row[4 * i + 0];
            }
            tmp.CopyTo(row);
        }

        private static void ShuffleInverseAvx512(Span<byte> row, int width)
        {
            if (!Avx512Vbmi.IsSupported) { ShuffleInverseAvx2(row, width); return; }

            var tmp = new byte[width * 4];
            int i = 0;

            var permInvArr = new byte[64];
            for (int fi = 0; fi < 16; fi++)
            {
                permInvArr[fi * 4 + 0] = (byte)(48 + fi);
                permInvArr[fi * 4 + 1] = (byte)(32 + fi);
                permInvArr[fi * 4 + 2] = (byte)(16 + fi);
                permInvArr[fi * 4 + 3] = (byte)( 0 + fi);
            }
            var perm0 = Unsafe.ReadUnaligned<Vector512<byte>>(
                ref MemoryMarshal.GetArrayDataReference(permInvArr));

            ref byte srcRef = ref MemoryMarshal.GetReference(row);
            ref byte dstRef = ref MemoryMarshal.GetArrayDataReference(tmp);

            for (; i <= width - 16; i += 16)
            {
                var p0 = Vector128.LoadUnsafe(ref Unsafe.Add(ref srcRef,             i));
                var p1 = Vector128.LoadUnsafe(ref Unsafe.Add(ref srcRef,     width + i));
                var p2 = Vector128.LoadUnsafe(ref Unsafe.Add(ref srcRef, 2 * width + i));
                var p3 = Vector128.LoadUnsafe(ref Unsafe.Add(ref srcRef, 3 * width + i));

                var lo  = Vector256.Create(p0, p1);
                var hi  = Vector256.Create(p2, p3);
                var src = Vector512.Create(lo, hi);

                var result = Avx512Vbmi.PermuteVar64x8(src, perm0);
                result.GetLower().GetLower().StoreUnsafe(ref Unsafe.Add(ref dstRef, i * 4));
                result.GetLower().GetUpper().StoreUnsafe(ref Unsafe.Add(ref dstRef, i * 4 + 16));
                result.GetUpper().GetLower().StoreUnsafe(ref Unsafe.Add(ref dstRef, i * 4 + 32));
                result.GetUpper().GetUpper().StoreUnsafe(ref Unsafe.Add(ref dstRef, i * 4 + 48));
            }

            for (; i < width; i++)
            {
                tmp[4 * i + 3] = row[            i];
                tmp[4 * i + 2] = row[    width + i];
                tmp[4 * i + 1] = row[2 * width + i];
                tmp[4 * i + 0] = row[3 * width + i];
            }
            tmp.CopyTo(row);
        }

        // ── ARM NEON (4 floats = 16 bytes per iteration) ───────────────────────

        private static void ShuffleForwardNeon(Span<byte> row, int width)
        {
            if (!AdvSimd.Arm64.IsSupported) { ShuffleForwardScalar(row, width); return; }

            var tmp = new byte[width * 4];
            int i = 0;

            ref byte srcRef = ref MemoryMarshal.GetReference(row);
            ref byte dstRef = ref MemoryMarshal.GetArrayDataReference(tmp);

            for (; i <= width - 4; i += 4)
            {
                var v = Vector128.LoadUnsafe(ref Unsafe.Add(ref srcRef, i * 4));

                var ue1 = AdvSimd.Arm64.UnzipEven(v, v);
                var uo1 = AdvSimd.Arm64.UnzipOdd(v, v);

                var ue2 = AdvSimd.Arm64.UnzipEven(ue1, uo1);
                var uo2 = AdvSimd.Arm64.UnzipOdd(ue1, uo1);

                var lsbPlane      = ue2.GetLower();
                var midLowPlane   = ue2.GetUpper();
                var midHighPlane  = uo2.GetLower();
                var msbPlane      = uo2.GetUpper();

                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef,             i), msbPlane.AsUInt32().GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef,     width + i), midHighPlane.AsUInt32().GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef, 2 * width + i), midLowPlane.AsUInt32().GetElement(0));
                Unsafe.WriteUnaligned(ref Unsafe.Add(ref dstRef, 3 * width + i), lsbPlane.AsUInt32().GetElement(0));
            }

            for (; i < width; i++)
            {
                tmp[i] = row[4 * i + 3];
                tmp[width + i] = row[4 * i + 2];
                tmp[2 * width + i] = row[4 * i + 1];
                tmp[3 * width + i] = row[4 * i + 0];
            }
            tmp.CopyTo(row);
        }

        private static void ShuffleInverseNeon(Span<byte> row, int width)
        {
            if (!AdvSimd.Arm64.IsSupported) { ShuffleInverseScalar(row, width); return; }

            var tmp = new byte[width * 4];
            int i = 0;

            ref byte srcRef = ref MemoryMarshal.GetReference(row);
            ref byte dstRef = ref MemoryMarshal.GetArrayDataReference(tmp);

            for (; i <= width - 4; i += 4)
            {
                var p0 = Vector64.CreateScalar(Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef,             i))).AsByte();
                var p1 = Vector64.CreateScalar(Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef,     width + i))).AsByte();
                var p2 = Vector64.CreateScalar(Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef, 2 * width + i))).AsByte();
                var p3 = Vector64.CreateScalar(Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref srcRef, 3 * width + i))).AsByte();

                var z32lo = AdvSimd.Arm64.ZipLow(p3, p2);
                var z10lo = AdvSimd.Arm64.ZipLow(p1, p0);

                var resLo = AdvSimd.Arm64.ZipLow( z32lo.AsUInt16(), z10lo.AsUInt16()).AsByte();
                var resHi = AdvSimd.Arm64.ZipHigh(z32lo.AsUInt16(), z10lo.AsUInt16()).AsByte();

                Vector128.Create(resLo, resHi).StoreUnsafe(ref Unsafe.Add(ref dstRef, i * 4));
            }

            for (; i < width; i++)
            {
                tmp[4 * i + 3] = row[            i];
                tmp[4 * i + 2] = row[    width + i];
                tmp[4 * i + 1] = row[2 * width + i];
                tmp[4 * i + 0] = row[3 * width + i];
            }
            tmp.CopyTo(row);
        }
    }
}
