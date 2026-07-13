using DZen.FloatingPointPredictor;
using System.Runtime.InteropServices;
using Xunit;

namespace DZen.FloatingPointPredictor.Tests
{
    // ═══════════════════════════════════════════════════════════════════════════
    // ── Fp32Predictor unit tests ─────────────────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // Tests are grouped into five classes:
    //
    //   RoundTripTests      – Encode→Decode recovers the original bytes exactly,
    //                         across every interesting width and a range of row counts.
    //
    //   ConformanceTests    – Hard-coded byte-level assertions that the shuffle and
    //                         delta steps produce the values specified by TIFF TN3.
    //
    //   RowIndependenceTests – Each row is processed independently; a tile with N
    //                          rows gives the same per-row result as N single-row
    //                          encodes.
    //
    //   SpecialValueTests   – NaN, ±Inf, ±0, subnormals, float.MaxValue all
    //                         survive the round-trip without bit corruption.
    //
    //   ScalarCorrectnessTests – Force the algorithm to run only its scalar path
    //                            by exercising widths of 1, 2, and 3 where no SIMD
    //                            path fires, then verify the same conformance values.
    // ═══════════════════════════════════════════════════════════════════════════

    // ── Helpers shared by all test classes ────────────────────────────────────

    internal static class TestHelpers
    {
        /// <summary>Converts a float array to a raw byte array (LE float32).</summary>
        public static byte[] FloatsToBytes(params float[] values)
        {
            var buf = new byte[values.Length * 4];
            MemoryMarshal.Cast<float, byte>(values).CopyTo(buf);
            return buf;
        }

        /// <summary>Converts a raw byte array back to floats.</summary>
        public static float[] BytesToFloats(byte[] bytes)
        {
            var floats = new float[bytes.Length / 4];
            MemoryMarshal.Cast<byte, float>(bytes.AsSpan()).CopyTo(floats);
            return floats;
        }

        /// <summary>
        /// Computes the expected encoded bytes for a single row the slow, obviously-
        /// correct way: byte-shuffle by hand then delta-encode each plane.
        /// This gives us a reference implementation independent of the SIMD paths.
        /// </summary>
        public static byte[] ReferenceEncode(float[] row)
        {
            int w = row.Length;
            var planar = new byte[w * 4];

            // Byte-shuffle: interleaved → 4 planes (MSB first)
            for (int i = 0; i < w; i++)
            {
                var fb = BitConverter.GetBytes(row[i]); // LE bytes: [b0, b1, b2, b3]
                planar[0 * w + i] = fb[3]; // plane 0 = MSB
                planar[1 * w + i] = fb[2]; // plane 1
                planar[2 * w + i] = fb[1]; // plane 2
                planar[3 * w + i] = fb[0]; // plane 3 = LSB
            }

            // Horizontal delta (right-to-left so we don't overwrite values we still need)
            for (int p = 0; p < 4; p++)
            {
                int off = p * w;
                for (int i = w - 1; i >= 1; i--)
                    planar[off + i] -= planar[off + i - 1];
            }

            return planar;
        }

        /// <summary>
        /// Decodes with the reference algorithm: undo delta then byte-unshuffle.
        /// </summary>
        public static float[] ReferenceDecode(byte[] encoded, int width)
        {
            var planar = (byte[])encoded.Clone();

            // Undo delta (left-to-right cumulative sum)
            for (int p = 0; p < 4; p++)
            {
                int off = p * width;
                for (int i = 1; i < width; i++)
                    planar[off + i] += planar[off + i - 1];
            }

            // Byte-unshuffle: 4 planes → interleaved
            var result = new float[width];
            for (int i = 0; i < width; i++)
            {
                byte b3 = planar[0 * width + i]; // MSB
                byte b2 = planar[1 * width + i];
                byte b1 = planar[2 * width + i];
                byte b0 = planar[3 * width + i]; // LSB
                result[i] = BitConverter.ToSingle(new byte[] { b0, b1, b2, b3 }, 0);
            }
            return result;
        }

        /// <summary>Produces a deterministic pseudo-random float array of length n.</summary>
        public static float[] RandomFloats(int n, int seed = 42)
        {
            var rng = new Random(seed);
            var arr = new float[n];
            for (int i = 0; i < n; i++)
                arr[i] = (float)(rng.NextDouble() * 200.0 - 100.0);
            return arr;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. Round-trip tests
    // ═══════════════════════════════════════════════════════════════════════════

    public class RoundTripTests
    {
        // Covers scalar tail, SSE chunk boundary, AVX2 boundary, AVX-512 boundary
        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(4)]
        [InlineData(5)]
        [InlineData(7)]
        [InlineData(8)]
        [InlineData(9)]
        [InlineData(15)]
        [InlineData(16)]
        [InlineData(17)]
        [InlineData(31)]
        [InlineData(32)]
        [InlineData(33)]
        [InlineData(63)]
        [InlineData(64)]
        [InlineData(127)]
        [InlineData(128)]
        [InlineData(256)]
        [InlineData(512)]
        public void SingleRow_RoundTrip(int width)
        {
            var original = TestHelpers.RandomFloats(width);
            var buf = TestHelpers.FloatsToBytes(original);

            Fp32Predictor.Encode(buf, width, rows: 1);
            Fp32Predictor.Decode(buf, width, rows: 1);

            var recovered = TestHelpers.BytesToFloats(buf);
            for (int i = 0; i < width; i++)
                Assert.Equal(original[i], recovered[i]); // exact bit equality
        }

        [Theory]
        [InlineData(1, 1)]
        [InlineData(4, 4)]
        [InlineData(8, 8)]
        [InlineData(16, 16)]
        [InlineData(512, 512)]
        [InlineData(512, 1)]
        [InlineData(1, 512)]
        [InlineData(7, 13)]
        [InlineData(33, 17)]
        public void MultiRow_RoundTrip(int width, int rows)
        {
            var original = TestHelpers.RandomFloats(width * rows, seed: width ^ rows);
            var buf = TestHelpers.FloatsToBytes(original);

            Fp32Predictor.Encode(buf, width, rows);
            Fp32Predictor.Decode(buf, width, rows);

            var recovered = TestHelpers.BytesToFloats(buf);
            for (int i = 0; i < original.Length; i++)
                Assert.Equal(original[i], recovered[i]);
        }

        [Fact]
        public void StandardCogTile_512x512_RoundTrip()
        {
            // A full 512×512 COG tile — the most common real-world tile size
            int width = 512, rows = 512;
            var original = TestHelpers.RandomFloats(width * rows, seed: 99);
            var buf = TestHelpers.FloatsToBytes(original);

            Fp32Predictor.Encode(buf, width, rows);
            Fp32Predictor.Decode(buf, width, rows);

            var recovered = TestHelpers.BytesToFloats(buf);
            for (int i = 0; i < original.Length; i++)
                Assert.Equal(original[i], recovered[i]);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. Conformance tests (exact byte values per TIFF TN3)
    // ═══════════════════════════════════════════════════════════════════════════

    public class ConformanceTests
    {
        // ── Byte-plane layout ──────────────────────────────────────────────────

        [Fact]
        public void Encode_SingleFloat_BytePlanesAreMsbFirst()
        {
            // 1.0f in IEEE 754 LE = bytes [0x00, 0x00, 0x80, 0x3F]
            //   b0=0x00  b1=0x00  b2=0x80  b3=0x3F
            // After shuffle (width=1 → no delta change):
            //   plane0[0] = b3 = 0x3F
            //   plane1[0] = b2 = 0x80
            //   plane2[0] = b1 = 0x00
            //   plane3[0] = b0 = 0x00
            var buf = TestHelpers.FloatsToBytes(1.0f);
            Fp32Predictor.Encode(buf, width: 1, rows: 1);

            Assert.Equal(0x3F, buf[0]); // plane 0 (MSB)
            Assert.Equal(0x80, buf[1]); // plane 1
            Assert.Equal(0x00, buf[2]); // plane 2
            Assert.Equal(0x00, buf[3]); // plane 3 (LSB)
        }

        [Fact]
        public void Encode_SingleFloat_MatchesReferenceImplementation()
        {
            float[] floats = { 1.0f };
            var expected = TestHelpers.ReferenceEncode(floats);
            var actual   = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(actual, width: 1, rows: 1);
            Assert.Equal(expected, actual);
        }

        // ── Delta encoding ─────────────────────────────────────────────────────

        [Fact]
        public void Encode_TwoFloats_DeltaEncodesEachPlane()
        {
            // 1.0f = [0x00, 0x00, 0x80, 0x3F]
            // 2.0f = [0x00, 0x00, 0x00, 0x40]
            //
            // After shuffle:
            //   plane0 = [0x3F, 0x40]   plane1 = [0x80, 0x00]
            //   plane2 = [0x00, 0x00]   plane3 = [0x00, 0x00]
            //
            // After delta (p[1] -= p[0]):
            //   plane0 = [0x3F, 0x01]   plane1 = [0x80, 0x80]  (0x00-0x80 wraps to 0x80)
            //   plane2 = [0x00, 0x00]   plane3 = [0x00, 0x00]
            var buf = TestHelpers.FloatsToBytes(1.0f, 2.0f);
            Fp32Predictor.Encode(buf, width: 2, rows: 1);

            Assert.Equal(new byte[] { 0x3F, 0x01, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00 }, buf);
        }

        [Fact]
        public void Encode_TwoFloats_MatchesReferenceImplementation()
        {
            float[] floats = { 1.0f, 2.0f };
            var expected = TestHelpers.ReferenceEncode(floats);
            var actual   = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(actual, width: 2, rows: 1);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Encode_FirstSampleOfEachPlane_IsNeverDeltaEncoded()
        {
            // The first sample in every plane carries the raw shuffled byte —
            // not a delta — regardless of width.
            float[] floats = TestHelpers.RandomFloats(32, seed: 7);
            var buf = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(buf, width: 32, rows: 1);

            // The un-delta'd first byte of each plane equals float[0]'s raw bytes.
            var f0 = BitConverter.GetBytes(floats[0]);
            Assert.Equal(f0[3], buf[0 * 32]); // plane 0 first byte = float[0] MSB
            Assert.Equal(f0[2], buf[1 * 32]); // plane 1
            Assert.Equal(f0[1], buf[2 * 32]); // plane 2
            Assert.Equal(f0[0], buf[3 * 32]); // plane 3 first byte = float[0] LSB
        }

        // ── Multi-width conformance against reference ──────────────────────────

        [Theory]
        [InlineData(1,  1)]
        [InlineData(4,  1)]
        [InlineData(5,  1)]   // 4 SIMD + 1 scalar tail
        [InlineData(8,  1)]
        [InlineData(9,  1)]
        [InlineData(16, 1)]
        [InlineData(17, 1)]
        [InlineData(32, 1)]
        [InlineData(64, 1)]
        public void Encode_MatchesReferenceImplementation(int width, int rows)
        {
            float[] floats = TestHelpers.RandomFloats(width * rows, seed: width + rows * 1000);
            var expected = new byte[floats.Length * 4];
            int rowBytes = width * 4;
            for (int r = 0; r < rows; r++)
            {
                var rowFloats = floats[(r * width)..((r + 1) * width)];
                var enc = TestHelpers.ReferenceEncode(rowFloats);
                enc.CopyTo(expected, r * rowBytes);
            }

            var actual = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(actual, width, rows);

            Assert.Equal(expected, actual);
        }

        // ── Decode is exact inverse of encode ─────────────────────────────────

        [Theory]
        [InlineData(1)]
        [InlineData(4)]
        [InlineData(8)]
        [InlineData(16)]
        [InlineData(32)]
        public void Decode_IsExactInverseOfEncode(int width)
        {
            float[] floats = TestHelpers.RandomFloats(width, seed: width * 17);
            var encoded = TestHelpers.ReferenceEncode(floats);
            var decoded = TestHelpers.ReferenceDecode(encoded, width);
            Assert.Equal(floats, decoded);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. Row independence tests
    // ═══════════════════════════════════════════════════════════════════════════

    public class RowIndependenceTests
    {
        /// <summary>
        /// Encoding each row of a multi-row tile must give identical bytes to
        /// encoding each row individually — the transform is row-local.
        /// </summary>
        [Theory]
        [InlineData(4,  3)]
        [InlineData(8,  4)]
        [InlineData(16, 8)]
        [InlineData(32, 2)]
        [InlineData(64, 5)]
        public void MultiRowEncode_EqualsIndividualRowEncodes(int width, int rows)
        {
            float[] data = TestHelpers.RandomFloats(width * rows, seed: width * rows);
            int rowBytes = width * 4;

            // Encode all rows together
            var bulk = TestHelpers.FloatsToBytes(data);
            Fp32Predictor.Encode(bulk, width, rows);

            // Encode each row separately
            var perRow = TestHelpers.FloatsToBytes(data);
            for (int r = 0; r < rows; r++)
                Fp32Predictor.Encode(perRow.AsSpan(r * rowBytes, rowBytes), width, rows: 1);

            Assert.Equal(bulk, perRow);
        }

        [Fact]
        public void DeltaDoesNotCarryAcrossRowBoundary()
        {
            // If delta carried across rows, row 1's first encoded byte would differ
            // depending on the last byte of row 0.  Build two tiles with identical
            // row 1 but different row 0 and verify row 1 encodes the same.
            int width = 8;
            float[] rowA0 = TestHelpers.RandomFloats(width, seed: 1);
            float[] rowA1 = TestHelpers.RandomFloats(width, seed: 2);
            float[] rowB0 = TestHelpers.RandomFloats(width, seed: 3); // different row 0
            float[] rowB1 = rowA1;                                     // same   row 1

            var bufA = TestHelpers.FloatsToBytes([.. rowA0, .. rowA1]);
            var bufB = TestHelpers.FloatsToBytes([.. rowB0, .. rowB1]);
            Fp32Predictor.Encode(bufA, width, rows: 2);
            Fp32Predictor.Encode(bufB, width, rows: 2);

            // Row 1's encoded bytes must be identical between bufA and bufB
            Assert.Equal(bufA[(width * 4)..], bufB[(width * 4)..]);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. Special value tests
    // ═══════════════════════════════════════════════════════════════════════════

    public class SpecialValueTests
    {
        [Fact]
        public void AllZeros_RoundTrip()
        {
            var buf = new byte[16 * 4]; // 16 zeros
            Fp32Predictor.Encode(buf, width: 16, rows: 1);
            Fp32Predictor.Decode(buf, width: 16, rows: 1);
            Assert.All(buf, b => Assert.Equal(0, b));
        }

        [Fact]
        public void AllZeros_EncodeIsAllZeros()
        {
            // 0.0f in IEEE 754 = 0x00000000; shuffle and delta both produce zeros.
            var buf = new byte[8 * 4];
            Fp32Predictor.Encode(buf, width: 8, rows: 1);
            Assert.All(buf, b => Assert.Equal(0, b));
        }

        [Theory]
        [MemberData(nameof(SpecialFloats))]
        public void SpecialValues_RoundTrip(float value)
        {
            var original = Enumerable.Repeat(value, 8).ToArray();
            var buf = TestHelpers.FloatsToBytes(original);
            Fp32Predictor.Encode(buf, width: 8, rows: 1);
            Fp32Predictor.Decode(buf, width: 8, rows: 1);
            var recovered = TestHelpers.BytesToFloats(buf);

            // Use raw bit comparison so NaN ≡ NaN
            for (int i = 0; i < original.Length; i++)
                Assert.Equal(
                    BitConverter.SingleToUInt32Bits(original[i]),
                    BitConverter.SingleToUInt32Bits(recovered[i]));
        }

        [Theory]
        [MemberData(nameof(SpecialFloats))]
        public void SpecialValues_EncodeMatchesReference(float value)
        {
            float[] floats = Enumerable.Repeat(value, 8).ToArray();
            var expected = TestHelpers.ReferenceEncode(floats);
            var actual   = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(actual, width: 8, rows: 1);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void MixedSpecialAndNormalValues_RoundTrip()
        {
            float[] floats =
            {
                0.0f, -0.0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity,
                float.MaxValue, float.MinValue, float.Epsilon,
                1.0f, -1.0f, 3.14159f, -2.71828f,
                float.NaN, 0.0f, 1.0f, 2.0f  // width=16 for full AVX2 coverage
            };
            var buf = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(buf, width: 16, rows: 1);
            Fp32Predictor.Decode(buf, width: 16, rows: 1);
            var recovered = TestHelpers.BytesToFloats(buf);

            for (int i = 0; i < floats.Length; i++)
                Assert.Equal(
                    BitConverter.SingleToUInt32Bits(floats[i]),
                    BitConverter.SingleToUInt32Bits(recovered[i]));
        }

        public static IEnumerable<object[]> SpecialFloats() =>
        [
            [float.NaN],
            [float.PositiveInfinity],
            [float.NegativeInfinity],
            [float.MaxValue],
            [float.MinValue],
            [float.Epsilon],
            [0.0f],
            [-0.0f],
        ];
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 5. Scalar-path correctness (widths 1, 2, 3 never hit any SIMD path)
    // ═══════════════════════════════════════════════════════════════════════════

    public class ScalarCorrectnessTests
    {
        [Fact]
        public void Width1_EncodeDecodeRoundTrip()
        {
            float[] floats = { 3.14159f };
            var buf = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(buf, width: 1, rows: 1);
            Fp32Predictor.Decode(buf, width: 1, rows: 1);
            Assert.Equal(floats[0], TestHelpers.BytesToFloats(buf)[0]);
        }

        [Fact]
        public void Width2_EncodeDecodeRoundTrip()
        {
            float[] floats = { 1.0f, 2.0f };
            var buf = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(buf, width: 2, rows: 1);
            Fp32Predictor.Decode(buf, width: 2, rows: 1);
            var recovered = TestHelpers.BytesToFloats(buf);
            Assert.Equal(floats[0], recovered[0]);
            Assert.Equal(floats[1], recovered[1]);
        }

        [Fact]
        public void Width3_EncodeDecodeRoundTrip()
        {
            float[] floats = { -1.5f, 0.0f, 1.5f };
            var buf = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(buf, width: 3, rows: 1);
            Fp32Predictor.Decode(buf, width: 3, rows: 1);
            var recovered = TestHelpers.BytesToFloats(buf);
            for (int i = 0; i < floats.Length; i++)
                Assert.Equal(floats[i], recovered[i]);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        public void ScalarWidth_EncodeMatchesReference(int width)
        {
            float[] floats = TestHelpers.RandomFloats(width, seed: width * 99);
            var expected = TestHelpers.ReferenceEncode(floats);
            var actual   = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(actual, width, rows: 1);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Width1_MultipleRows_EachRowIsIndependent()
        {
            // With width=1 there is no delta (single sample = delta of 0), so
            // encode is purely a byte shuffle.  Four rows of different values
            // must each encode to the shuffled bytes of that float alone.
            float[] floats = { 1.0f, 2.0f, 3.0f, 4.0f };
            var buf = TestHelpers.FloatsToBytes(floats);
            Fp32Predictor.Encode(buf, width: 1, rows: 4);

            for (int r = 0; r < 4; r++)
            {
                var fb = BitConverter.GetBytes(floats[r]);
                Assert.Equal(fb[3], buf[r * 4 + 0]); // plane 0 MSB
                Assert.Equal(fb[2], buf[r * 4 + 1]); // plane 1
                Assert.Equal(fb[1], buf[r * 4 + 2]); // plane 2
                Assert.Equal(fb[0], buf[r * 4 + 3]); // plane 3 LSB
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 6. API contract tests
    // ═══════════════════════════════════════════════════════════════════════════

    public class ApiContractTests
    {
        [Fact]
        public void Encode_EmptyTile_DoesNotThrow()
        {
            // rows=0 or width=0 — nothing to do, must not throw
            var buf = Array.Empty<byte>();
            Fp32Predictor.Encode(buf, width: 0, rows: 0);
        }

        [Fact]
        public void Decode_EmptyTile_DoesNotThrow()
        {
            var buf = Array.Empty<byte>();
            Fp32Predictor.Decode(buf, width: 0, rows: 0);
        }

        [Fact]
        public void Encode_IsInPlace_ReturnsVoid()
        {
            // The method must modify its input span in-place (no allocation returned)
            float[] f = { 1.0f, 2.0f, 3.0f, 4.0f };
            var buf = TestHelpers.FloatsToBytes(f);
            var span = buf.AsSpan();
            Fp32Predictor.Encode(span, width: 4, rows: 1);
            // buf should now differ from the original float bytes
            Assert.NotEqual(TestHelpers.FloatsToBytes(f), buf);
        }

        [Fact]
        public void Decode_IsInPlace_ReturnsVoid()
        {
            float[] f = { 1.0f, 2.0f, 3.0f, 4.0f };
            var buf = TestHelpers.FloatsToBytes(f);
            Fp32Predictor.Encode(buf, width: 4, rows: 1);
            var encoded = (byte[])buf.Clone();
            Fp32Predictor.Decode(buf, width: 4, rows: 1);
            Assert.NotEqual(encoded, buf);              // decode changed the buffer
            Assert.Equal(TestHelpers.FloatsToBytes(f), buf); // and got back original
        }

        [Fact]
        public void EncodeIsIdempotentAfterDecodeEncode()
        {
            // Encode(Decode(Encode(x))) == Encode(x)
            float[] f = TestHelpers.RandomFloats(16, seed: 1234);
            var buf = TestHelpers.FloatsToBytes(f);

            Fp32Predictor.Encode(buf, width: 16, rows: 1);
            var firstEncode = (byte[])buf.Clone();

            Fp32Predictor.Decode(buf, width: 16, rows: 1);
            Fp32Predictor.Encode(buf, width: 16, rows: 1);

            Assert.Equal(firstEncode, buf);
        }

        [Fact]
        public void Encode_UndersizedTile_ThrowsBeforeMutation()
        {
            byte[] buf = Enumerable.Range(1, 31).Select(i => (byte)i).ToArray();
            var original = (byte[])buf.Clone();

            Assert.Throws<ArgumentException>(() => Fp32Predictor.Encode(buf, width: 4, rows: 2));
            Assert.Equal(original, buf);
        }

        [Theory]
        [InlineData(-1, 1)]
        [InlineData(1, -1)]
        public void Encode_NegativeDimensions_Throw(int width, int rows)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                Fp32Predictor.Encode(Array.Empty<byte>(), width, rows));
        }
    }
}
