using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;

namespace DZen.FloatingPointPredictor
{
    // ── Horizontal Differencing Predictor (TIFF Predictor = 2) ─────────────────

    /// <summary>
    /// Implements the horizontal differencing predictor defined by TIFF Predictor = 2.
    ///
    /// Algorithm overview
    /// ──────────────────
    /// Each sample in a row is replaced by the difference between that sample and
    /// the preceding one.  For multi-byte samples, the subtraction is performed
    /// as a complete little-endian unsigned value, with arithmetic wrapping at
    /// the sample width:
    ///
    ///   s[i] ← s[i] − s[i−1]   (encode, right-to-left)
    ///   s[i] ← s[i] + s[i−1]   (decode, left-to-right)
    ///
    /// The first sample of every row is left unchanged — it carries the raw value.
    ///
    /// Hardware dispatch
    /// ─────────────────
    /// When <c>bytesPerSample == 1</c> the inner loop is byte-level subtraction
    /// with stride 1, which maps naturally to SIMD:
    ///
    ///   • AVX-512BW  (64 bytes per iteration)
    ///   • AVX2       (32 bytes per iteration)
    ///   • SSSE3/SSE2 (16 bytes per iteration)
    ///   • ARM NEON   (16 bytes per iteration)
    ///   • Scalar     (portable, no SIMD)
    ///
    /// For <c>bytesPerSample &gt; 1</c> only the scalar path is used.
    ///
    /// Thread safety: all public methods are stateless and thread-safe.
    /// </summary>
    public static class BytePredictor
    {
        // ── Public API ─────────────────────────────────────────────────────────

        /// <summary>
        /// Applies TIFF horizontal differencing (Predictor=2) encode in-place to a
        /// single tile (or strip) stored in <paramref name="tile"/>.
        ///
        /// The tile is treated as <paramref name="rows"/> × <paramref name="width"/>
        /// samples of <paramref name="bytesPerSample"/> bytes each (total bytes =
        /// rows × width × bytesPerSample).  Each row is processed independently.
        /// </summary>
        /// <param name="tile">Raw tile bytes in row-major order.</param>
        /// <param name="width">Number of samples per row.</param>
        /// <param name="rows">Number of rows in the tile.</param>
        /// <param name="bytesPerSample">Number of bytes per sample (default 1).</param>
        public static void Encode(Span<byte> tile, int width, int rows, int bytesPerSample = 1)
        {
            int rowBytes = ValidateArguments(tile.Length, width, rows, bytesPerSample);
            for (int r = 0; r < rows; r++)
                EncodeRow(tile.Slice(r * rowBytes, rowBytes), width, bytesPerSample);
        }

        /// <summary>
        /// Reverses the TIFF horizontal differencing (Predictor=2) transform in-place.
        /// Operates on each row independently; each row must be exactly
        /// <paramref name="width"/> × <paramref name="bytesPerSample"/> bytes.
        /// </summary>
        public static void Decode(Span<byte> tile, int width, int rows, int bytesPerSample = 1)
        {
            int rowBytes = ValidateArguments(tile.Length, width, rows, bytesPerSample);
            for (int r = 0; r < rows; r++)
                DecodeRow(tile.Slice(r * rowBytes, rowBytes), width, bytesPerSample);
        }

        /// <summary>
        /// Decodes multi-byte data produced by DZen.FloatingPointPredictor 1.x,
        /// which applied Predictor=2 arithmetic independently to each byte.
        /// </summary>
        /// <remarks>
        /// Use this method only for legacy payloads written by version 1.x with
        /// <paramref name="bytesPerSample"/> greater than 1. New and migrated data
        /// must use <see cref="Decode(Span{byte}, int, int, int)"/>.
        /// </remarks>
        public static void DecodeLegacyBytewise(Span<byte> tile, int width, int rows, int bytesPerSample = 1)
        {
            int rowBytes = ValidateArguments(tile.Length, width, rows, bytesPerSample);
            for (int r = 0; r < rows; r++)
                DecodeLegacyBytewiseRow(tile.Slice(r * rowBytes, rowBytes), width, bytesPerSample);
        }

        private static int ValidateArguments(int tileLength, int width, int rows, int bytesPerSample)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(width);
            ArgumentOutOfRangeException.ThrowIfNegative(rows);
            ArgumentOutOfRangeException.ThrowIfLessThan(bytesPerSample, 1);

            if (width == 0 || rows == 0)
                return 0;

            long rowBytes = (long)width * bytesPerSample;
            if (rowBytes > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(width), "A row exceeds the maximum supported span length.");

            long requiredBytes = rowBytes * rows;
            if (requiredBytes > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(rows), "The tile dimensions exceed the maximum supported span length.");
            if (tileLength < requiredBytes)
                throw new ArgumentException("The tile is shorter than the supplied dimensions require.", "tile");

            return (int)rowBytes;
        }

        // ── Per-row entry points ───────────────────────────────────────────────

        private static void EncodeRow(Span<byte> row, int width, int bps)
        {
            int totalBytes = width * bps;
            if (totalBytes <= bps) return;

            if (bps == 1)
                DeltaEncodeScalar(row, totalBytes);
            else
                DeltaEncodeStrided(row, width, bps);
        }

        private static void DecodeRow(Span<byte> row, int width, int bps)
        {
            int totalBytes = width * bps;
            if (totalBytes <= bps) return;

            if (bps == 1)
                DeltaDecodeScalar(row, totalBytes);
            else
                DeltaDecodeStrided(row, width, bps);
        }

        private static void DecodeLegacyBytewiseRow(Span<byte> row, int width, int bps)
        {
            int totalBytes = width * bps;
            if (totalBytes <= bps) return;

            if (bps == 1)
                DeltaDecodeScalar(row, totalBytes);
            else
                DeltaDecodeLegacyBytewise(row, width, bps);
        }

        // ── Scalar implementations ─────────────────────────────────────────────

        private static void DeltaEncodeStrided(Span<byte> row, int width, int bps)
        {
            for (int i = width - 1; i >= 1; i--)
            {
                int cur = i * bps;
                int prev = cur - bps;
                int borrow = 0;
                for (int b = 0; b < bps; b++)
                {
                    int difference = row[cur + b] - row[prev + b] - borrow;
                    row[cur + b] = unchecked((byte)difference);
                    borrow = difference < 0 ? 1 : 0;
                }
            }
        }

        private static void DeltaDecodeStrided(Span<byte> row, int width, int bps)
        {
            for (int i = 1; i < width; i++)
            {
                int cur = i * bps;
                int prev = cur - bps;
                int carry = 0;
                for (int b = 0; b < bps; b++)
                {
                    int sum = row[cur + b] + row[prev + b] + carry;
                    row[cur + b] = unchecked((byte)sum);
                    carry = sum >> 8;
                }
            }
        }

        private static void DeltaDecodeLegacyBytewise(Span<byte> row, int width, int bps)
        {
            for (int i = 1; i < width; i++)
            {
                int cur = i * bps;
                int prev = cur - bps;
                for (int b = 0; b < bps; b++)
                    row[cur + b] += row[prev + b];
            }
        }

        // ── Scalar dispatch for bps=1 (delegates to fastest SIMD path) ────────

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void DeltaEncodeScalar(Span<byte> row, int totalBytes)
        {
            if (Avx512BW.IsSupported && totalBytes > 64)
                DeltaEncodeAvx512(row, totalBytes);
            else if (Avx2.IsSupported && totalBytes > 32)
                DeltaEncodeAvx2(row, totalBytes);
            else if (Sse2.IsSupported && totalBytes > 16)
                DeltaEncodeSse2(row, totalBytes);
            else if (AdvSimd.IsSupported && totalBytes > 16)
                DeltaEncodeNeon(row, totalBytes);
            else
                DeltaEncodePureScalar(row, totalBytes);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void DeltaDecodeScalar(Span<byte> row, int totalBytes)
        {
            if (Avx512BW.IsSupported && totalBytes >= 64)
                DeltaDecodeAvx512(row, totalBytes);
            else if (Avx2.IsSupported && totalBytes >= 32)
                DeltaDecodeAvx2(row, totalBytes);
            else if (Sse2.IsSupported && totalBytes >= 16)
                DeltaDecodeSse2(row, totalBytes);
            else if (AdvSimd.IsSupported && totalBytes >= 16)
                DeltaDecodeNeon(row, totalBytes);
            else
                DeltaDecodePureScalar(row, totalBytes);
        }

        private static void DeltaEncodePureScalar(Span<byte> row, int totalBytes)
        {
            for (int i = totalBytes - 1; i >= 1; i--)
                row[i] -= row[i - 1];
        }

        private static void DeltaDecodePureScalar(Span<byte> row, int totalBytes)
        {
            for (int i = 1; i < totalBytes; i++)
                row[i] += row[i - 1];
        }

        // ── SSE2 encode (16 bytes per iteration) ───────────────────────────────

        private static void DeltaEncodeSse2(Span<byte> row, int totalBytes)
        {
            if (!Sse2.IsSupported) { DeltaEncodePureScalar(row, totalBytes); return; }

            int i = totalBytes;
            ref byte rowRef = ref MemoryMarshal.GetReference(row);

            for (; i > 16; i -= 16)
            {
                var cur = Vector128.LoadUnsafe(ref Unsafe.Add(ref rowRef, i - 16));
                var prv = Vector128.LoadUnsafe(ref Unsafe.Add(ref rowRef, i - 17));
                var res = Sse2.Subtract(cur, prv);
                res.StoreUnsafe(ref Unsafe.Add(ref rowRef, i - 16));
            }

            for (; i >= 2; i--)
                row[i - 1] -= row[i - 2];
        }

        // ── SSE2 decode (16 bytes per iteration) ───────────────────────────────

        private static void DeltaDecodeSse2(Span<byte> row, int totalBytes)
        {
            if (!Sse2.IsSupported) { DeltaDecodePureScalar(row, totalBytes); return; }

            int i = 0;
            byte running = 0;
            ref byte rowRef = ref MemoryMarshal.GetReference(row);

            if (totalBytes >= 16)
            {
                var v0 = Vector128.LoadUnsafe(ref rowRef);
                var ps = PrefixSum16(v0);
                ps.StoreUnsafe(ref rowRef);
                running = Unsafe.Add(ref rowRef, 15);
                i = 16;
            }

            for (; i <= totalBytes - 16; i += 16)
            {
                ref byte chunkRef = ref Unsafe.Add(ref rowRef, i);
                var v = Vector128.LoadUnsafe(ref chunkRef);
                var ps = PrefixSum16(v);
                var bro = Vector128.Create(running);
                var res = Sse2.Add(ps, bro);
                res.StoreUnsafe(ref chunkRef);
                running = Unsafe.Add(ref rowRef, i + 15);
            }

            for (; i < totalBytes; i++)
                row[i] += row[i - 1];
        }

        // ── AVX2 encode (32 bytes per iteration) ───────────────────────────────

        private static void DeltaEncodeAvx2(Span<byte> row, int totalBytes)
        {
            if (!Avx2.IsSupported) { DeltaEncodeSse2(row, totalBytes); return; }

            int i = totalBytes;
            ref byte rowRef = ref MemoryMarshal.GetReference(row);

            for (; i > 32; i -= 32)
            {
                var cur = Vector256.LoadUnsafe(ref Unsafe.Add(ref rowRef, i - 32));
                var prv = Vector256.LoadUnsafe(ref Unsafe.Add(ref rowRef, i - 33));
                var res = Avx2.Subtract(cur, prv);
                res.StoreUnsafe(ref Unsafe.Add(ref rowRef, i - 32));
            }

            DeltaEncodeSse2(row[..i], i);
        }

        // ── AVX2 decode (32 bytes per iteration) ───────────────────────────────

        private static void DeltaDecodeAvx2(Span<byte> row, int totalBytes)
        {
            if (!Avx2.IsSupported) { DeltaDecodeSse2(row, totalBytes); return; }

            int i = 0;
            byte running = 0;
            ref byte rowRef = ref MemoryMarshal.GetReference(row);

            if (totalBytes >= 32)
            {
                var v0 = Vector256.LoadUnsafe(ref rowRef);
                var lo = PrefixSum16(v0.GetLower());
                var up = PrefixSum16(v0.GetUpper());
                byte lastLo = lo.GetElement(15);
                var bro = Vector128.Create(lastLo);
                var adj = Sse2.Add(up, bro);
                var res = Vector256.Create(lo, adj);
                res.StoreUnsafe(ref rowRef);
                running = Unsafe.Add(ref rowRef, 31);
                i = 32;
            }

            for (; i <= totalBytes - 32; i += 32)
            {
                ref byte chunkRef = ref Unsafe.Add(ref rowRef, i);
                var v = Vector256.LoadUnsafe(ref chunkRef);
                var lo = PrefixSum16(v.GetLower());
                var up = PrefixSum16(v.GetUpper());
                byte lastLo = lo.GetElement(15);
                var broLo = Vector128.Create(lastLo);
                var adjUp = Sse2.Add(up, broLo);
                var ps = Vector256.Create(lo, adjUp);
                var bro64 = Vector256.Create(running);
                var res = Avx2.Add(ps, bro64);
                res.StoreUnsafe(ref chunkRef);
                running = Unsafe.Add(ref rowRef, i + 31);
            }

            for (; i < totalBytes; i++)
                row[i] += row[i - 1];
        }

        // ── AVX-512BW encode (64 bytes per iteration) ──────────────────────────

        private static void DeltaEncodeAvx512(Span<byte> row, int totalBytes)
        {
            if (!Avx512BW.IsSupported) { DeltaEncodeAvx2(row, totalBytes); return; }

            int i = totalBytes;
            ref byte rowRef = ref MemoryMarshal.GetReference(row);

            for (; i > 64; i -= 64)
            {
                var cur = Vector512.LoadUnsafe(ref Unsafe.Add(ref rowRef, i - 64));
                var prv = Vector512.LoadUnsafe(ref Unsafe.Add(ref rowRef, i - 65));
                var res = Avx512BW.Subtract(cur, prv);
                res.StoreUnsafe(ref Unsafe.Add(ref rowRef, i - 64));
            }

            DeltaEncodeAvx2(row[..i], i);
        }

        // ── AVX-512BW decode (64 bytes per iteration) ──────────────────────────

        private static void DeltaDecodeAvx512(Span<byte> row, int totalBytes)
        {
            if (!Avx512BW.IsSupported) { DeltaDecodeAvx2(row, totalBytes); return; }

            int i = 0;
            byte running = 0;
            ref byte rowRef = ref MemoryMarshal.GetReference(row);

            if (totalBytes >= 64)
            {
                var v0 = Vector512.LoadUnsafe(ref rowRef);
                var l0 = PrefixSum16(v0.GetLower().GetLower());
                var l1 = PrefixSum16(v0.GetLower().GetUpper());
                var l2 = PrefixSum16(v0.GetUpper().GetLower());
                var l3 = PrefixSum16(v0.GetUpper().GetUpper());

                byte b0 = l0.GetElement(15);
                var l1adj = Sse2.Add(l1, Vector128.Create(b0));
                byte b1 = l1adj.GetElement(15);
                var l2adj = Sse2.Add(l2, Vector128.Create(b1));
                byte b2 = l2adj.GetElement(15);
                var l3adj = Sse2.Add(l3, Vector128.Create(b2));

                var res = Vector512.Create(
                    Vector256.Create(l0, l1adj),
                    Vector256.Create(l2adj, l3adj));
                res.StoreUnsafe(ref rowRef);
                running = Unsafe.Add(ref rowRef, 63);
                i = 64;
            }

            for (; i <= totalBytes - 64; i += 64)
            {
                ref byte chunkRef = ref Unsafe.Add(ref rowRef, i);
                var v = Vector512.LoadUnsafe(ref chunkRef);
                var l0 = PrefixSum16(v.GetLower().GetLower());
                var l1 = PrefixSum16(v.GetLower().GetUpper());
                var l2 = PrefixSum16(v.GetUpper().GetLower());
                var l3 = PrefixSum16(v.GetUpper().GetUpper());

                byte b0 = l0.GetElement(15);
                var l1adj = Sse2.Add(l1, Vector128.Create(b0));
                byte b1 = l1adj.GetElement(15);
                var l2adj = Sse2.Add(l2, Vector128.Create(b1));
                byte b2 = l2adj.GetElement(15);
                var l3adj = Sse2.Add(l3, Vector128.Create(b2));

                var ps = Vector512.Create(
                    Vector256.Create(l0, l1adj),
                    Vector256.Create(l2adj, l3adj));
                var bro = Vector512.Create(running);
                var res = Avx512BW.Add(ps, bro);
                res.StoreUnsafe(ref chunkRef);
                running = Unsafe.Add(ref rowRef, i + 63);
            }

            for (; i < totalBytes; i++)
                row[i] += row[i - 1];
        }

        // ── ARM NEON encode / decode (16 bytes per iteration) ──────────────────

        private static void DeltaEncodeNeon(Span<byte> row, int totalBytes)
        {
            if (!AdvSimd.IsSupported) { DeltaEncodePureScalar(row, totalBytes); return; }

            int i = totalBytes;
            ref byte rowRef = ref MemoryMarshal.GetReference(row);

            for (; i > 16; i -= 16)
            {
                var cur = Vector128.LoadUnsafe(ref Unsafe.Add(ref rowRef, i - 16));
                var prv = Vector128.LoadUnsafe(ref Unsafe.Add(ref rowRef, i - 17));
                var res = AdvSimd.Subtract(cur, prv);
                res.StoreUnsafe(ref Unsafe.Add(ref rowRef, i - 16));
            }

            for (; i >= 2; i--)
                row[i - 1] -= row[i - 2];
        }

        private static void DeltaDecodeNeon(Span<byte> row, int totalBytes)
        {
            if (!AdvSimd.IsSupported) { DeltaDecodePureScalar(row, totalBytes); return; }

            int i = 0;
            byte running = 0;
            ref byte rowRef = ref MemoryMarshal.GetReference(row);

            if (totalBytes >= 16)
            {
                var v0 = Vector128.LoadUnsafe(ref rowRef);
                var ps = PrefixSum16Neon(v0);
                ps.StoreUnsafe(ref rowRef);
                running = Unsafe.Add(ref rowRef, 15);
                i = 16;
            }

            for (; i <= totalBytes - 16; i += 16)
            {
                ref byte chunkRef = ref Unsafe.Add(ref rowRef, i);
                var v = Vector128.LoadUnsafe(ref chunkRef);
                var ps = PrefixSum16Neon(v);
                var bro = Vector128.Create(running);
                var res = AdvSimd.Add(ps, bro);
                res.StoreUnsafe(ref chunkRef);
                running = Unsafe.Add(ref rowRef, i + 15);
            }

            for (; i < totalBytes; i++)
                row[i] += row[i - 1];
        }

        // ── Prefix-sum kernels (16-byte exclusive → inclusive via shift+add) ───

        /// <summary>
        /// Computes the inclusive byte-wise prefix sum of a 16-byte vector.
        /// Algorithm: shift-and-add in log₂(16)=4 steps.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<byte> PrefixSum16(Vector128<byte> v)
        {
            if (Sse2.IsSupported)
            {
                var s1 = Sse2.Add(v, Sse2.ShiftLeftLogical128BitLane(v, 1));
                var s2 = Sse2.Add(s1, Sse2.ShiftLeftLogical128BitLane(s1, 2));
                var s4 = Sse2.Add(s2, Sse2.ShiftLeftLogical128BitLane(s2, 4));
                var s8 = Sse2.Add(s4, Sse2.ShiftLeftLogical128BitLane(s4, 8));
                return s8;
            }
            if (AdvSimd.IsSupported)
                return PrefixSum16Neon(v);
            return PrefixSum16Slow(v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<byte> PrefixSum16Neon(Vector128<byte> v)
        {
            Span<byte> buf = stackalloc byte[16];
            v.StoreUnsafe(ref MemoryMarshal.GetReference(buf));
            byte acc = 0;
            for (int i = 0; i < 16; i++)
            {
                acc += buf[i];
                buf[i] = acc;
            }
            return Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(buf));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<byte> PrefixSum16Slow(Vector128<byte> v)
        {
            Span<byte> buf = stackalloc byte[16];
            v.CopyTo(buf);
            byte acc = 0;
            for (int i = 0; i < 16; i++)
            {
                acc += buf[i];
                buf[i] = acc;
            }
            return Vector128.Create(
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
                buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]);
        }
    }
}
