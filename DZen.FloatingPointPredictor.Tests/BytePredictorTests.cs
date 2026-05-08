using DZen.FloatingPointPredictor;
using Xunit;

namespace DZen.FloatingPointPredictor.Tests
{
    // ═══════════════════════════════════════════════════════════════════════════
    // ── BytePredictor unit tests ───────────────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // Tests are grouped into seven classes:
    //
    //   RoundTripTests       – Encode→Decode recovers the original bytes exactly,
    //                          across every interesting width.
    //
    //   ConformanceTests     – Hard-coded byte-level assertions that the delta
    //                          produces the values specified by TIFF Predictor=2.
    //
    //   RowIndependenceTests – Each row is processed independently.
    //
    //   SpecialValueTests    – Edge cases: zeros, 0xFF wrapping, overflows.
    //
    //   ScalarCorrectnessTests – Force scalar path via small widths.
    //
    //   StrideTests          – bytesPerSample > 1 (2, 3, 4, 8).
    //
    //   ApiContractTests     – Documents observable API behavior.
    // ═══════════════════════════════════════════════════════════════════════════

    internal static class ByteTestHelpers
    {
        public static byte[] RandomBytes(int n, int seed = 42)
        {
            var rng = new Random(seed);
            var buf = new byte[n];
            rng.NextBytes(buf);
            return buf;
        }

        public static byte[] ReferenceEncodeStride1(byte[] row)
        {
            var enc = (byte[])row.Clone();
            for (int i = enc.Length - 1; i >= 1; i--)
                enc[i] -= enc[i - 1];
            return enc;
        }

        public static byte[] ReferenceDecodeStride1(byte[] encoded)
        {
            var dec = (byte[])encoded.Clone();
            for (int i = 1; i < dec.Length; i++)
                dec[i] += dec[i - 1];
            return dec;
        }

        public static byte[] ReferenceEncodeStrided(byte[] row, int width, int bps)
        {
            var enc = (byte[])row.Clone();
            for (int i = width - 1; i >= 1; i--)
            {
                int cur = i * bps;
                int prev = cur - bps;
                for (int b = 0; b < bps; b++)
                    enc[cur + b] -= enc[prev + b];
            }
            return enc;
        }

        public static byte[] ReferenceDecodeStrided(byte[] encoded, int width, int bps)
        {
            var dec = (byte[])encoded.Clone();
            for (int i = 1; i < width; i++)
            {
                int cur = i * bps;
                int prev = cur - bps;
                for (int b = 0; b < bps; b++)
                    dec[cur + b] += dec[prev + b];
            }
            return dec;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. Round-trip tests (bps=1)
    // ═══════════════════════════════════════════════════════════════════════════

    public class ByteRoundTripTests
    {
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
        [InlineData(65)]
        [InlineData(127)]
        [InlineData(128)]
        [InlineData(256)]
        [InlineData(512)]
        public void SingleRow_RoundTrip(int width)
        {
            var original = ByteTestHelpers.RandomBytes(width, seed: width);
            var buf = (byte[])original.Clone();

            BytePredictor.Encode(buf, width, rows: 1);
            BytePredictor.Decode(buf, width, rows: 1);

            Assert.Equal(original, buf);
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
            var original = ByteTestHelpers.RandomBytes(width * rows, seed: width ^ rows);
            var buf = (byte[])original.Clone();

            BytePredictor.Encode(buf, width, rows);
            BytePredictor.Decode(buf, width, rows);

            Assert.Equal(original, buf);
        }

        [Fact]
        public void LargeTile_512x512_RoundTrip()
        {
            int width = 512, rows = 512;
            var original = ByteTestHelpers.RandomBytes(width * rows, seed: 99);
            var buf = (byte[])original.Clone();

            BytePredictor.Encode(buf, width, rows);
            BytePredictor.Decode(buf, width, rows);

            Assert.Equal(original, buf);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. Conformance tests (exact byte values) (bps=1)
    // ═══════════════════════════════════════════════════════════════════════════

    public class ByteConformanceTests
    {
        [Fact]
        public void Encode_SingleByte_IsUnchanged()
        {
            var buf = new byte[] { 0x42 };
            BytePredictor.Encode(buf, width: 1, rows: 1);
            Assert.Equal(0x42, buf[0]);
        }

        [Fact]
        public void Encode_TwoBytes_DeltaEncoded()
        {
            // row = [0x10, 0x20]
            // encode: row[1] -= row[0] → 0x20 - 0x10 = 0x10
            // result: [0x10, 0x10]
            var buf = new byte[] { 0x10, 0x20 };
            BytePredictor.Encode(buf, width: 2, rows: 1);
            Assert.Equal(new byte[] { 0x10, 0x10 }, buf);
        }

        [Fact]
        public void Encode_ThreeBytesWithWrapping()
        {
            // row = [0x80, 0x10, 0x30]
            // encode right-to-left:
            //   i=2: buf[2] = 0x30 - 0x10 = 0x20
            //   i=1: buf[1] = 0x10 - 0x80 → wraps: 0x10 - 0x80 = 0x90 (256 - 128 + 16 = 144)
            // result: [0x80, 0x90, 0x20]
            var buf = new byte[] { 0x80, 0x10, 0x30 };
            BytePredictor.Encode(buf, width: 3, rows: 1);
            Assert.Equal(new byte[] { 0x80, 0x90, 0x20 }, buf);
        }

        [Fact]
        public void Encode_FirstSampleNeverDeltaEncoded()
        {
            var original = ByteTestHelpers.RandomBytes(64, seed: 7);
            var buf = (byte[])original.Clone();
            BytePredictor.Encode(buf, width: 64, rows: 1);
            Assert.Equal(original[0], buf[0]);
        }

        [Fact]
        public void Encode_MatchesReferenceImplementation()
        {
            byte[] row = ByteTestHelpers.RandomBytes(64, seed: 123);
            var expected = ByteTestHelpers.ReferenceEncodeStride1(row);
            var actual = (byte[])row.Clone();
            BytePredictor.Encode(actual, width: 64, rows: 1);
            Assert.Equal(expected, actual);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(4)]
        [InlineData(5)]
        [InlineData(8)]
        [InlineData(9)]
        [InlineData(16)]
        [InlineData(17)]
        [InlineData(32)]
        [InlineData(33)]
        [InlineData(64)]
        [InlineData(65)]
        public void Encode_MatchesReference_MultiWidth(int width)
        {
            byte[] row = ByteTestHelpers.RandomBytes(width, seed: width * 13);
            var expected = ByteTestHelpers.ReferenceEncodeStride1(row);
            var actual = (byte[])row.Clone();
            BytePredictor.Encode(actual, width, rows: 1);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Decode_IsExactInverseOfEncode()
        {
            byte[] row = ByteTestHelpers.RandomBytes(64, seed: 77);
            var encoded = ByteTestHelpers.ReferenceEncodeStride1(row);
            var decoded = ByteTestHelpers.ReferenceDecodeStride1(encoded);
            Assert.Equal(row, decoded);
        }

        [Fact]
        public void Decode_MatchesReferenceImplementation()
        {
            byte[] row = ByteTestHelpers.RandomBytes(64, seed: 88);
            var encoded = ByteTestHelpers.ReferenceEncodeStride1(row);
            var actual = (byte[])encoded.Clone();
            BytePredictor.Decode(actual, width: 64, rows: 1);
            Assert.Equal(row, actual);
        }

        // ── Multi-row conformance ──────────────────────────────────────────────

        [Theory]
        [InlineData(4, 3)]
        [InlineData(8, 4)]
        [InlineData(16, 2)]
        [InlineData(32, 3)]
        [InlineData(64, 5)]
        public void EncodeMultiRow_MatchesReferencePerRow(int width, int rows)
        {
            byte[] data = ByteTestHelpers.RandomBytes(width * rows, seed: width * rows);
            int rowBytes = width;
            var expected = new byte[data.Length];
            for (int r = 0; r < rows; r++)
            {
                var row = data[(r * rowBytes)..((r + 1) * rowBytes)];
                var enc = ByteTestHelpers.ReferenceEncodeStride1(row);
                enc.CopyTo(expected, r * rowBytes);
            }

            var actual = (byte[])data.Clone();
            BytePredictor.Encode(actual, width, rows);
            Assert.Equal(expected, actual);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. Row independence tests
    // ═══════════════════════════════════════════════════════════════════════════

    public class ByteRowIndependenceTests
    {
        [Theory]
        [InlineData(4, 3)]
        [InlineData(8, 4)]
        [InlineData(16, 8)]
        [InlineData(32, 2)]
        [InlineData(64, 5)]
        public void MultiRowEncode_EqualsIndividualRowEncodes(int width, int rows)
        {
            byte[] data = ByteTestHelpers.RandomBytes(width * rows, seed: width * rows);
            int rowBytes = width;

            var bulk = (byte[])data.Clone();
            BytePredictor.Encode(bulk, width, rows);

            var perRow = (byte[])data.Clone();
            for (int r = 0; r < rows; r++)
                BytePredictor.Encode(perRow.AsSpan(r * rowBytes, rowBytes), width, rows: 1);

            Assert.Equal(bulk, perRow);
        }

        [Fact]
        public void DeltaDoesNotCarryAcrossRowBoundary()
        {
            int width = 8;
            byte[] rowA0 = ByteTestHelpers.RandomBytes(width, seed: 1);
            byte[] rowA1 = ByteTestHelpers.RandomBytes(width, seed: 2);
            byte[] rowB0 = ByteTestHelpers.RandomBytes(width, seed: 3);
            byte[] rowB1 = rowA1;

            var bufA = (byte[])[.. rowA0, .. rowA1];
            var bufB = (byte[])[.. rowB0, .. rowB1];
            BytePredictor.Encode(bufA, width, rows: 2);
            BytePredictor.Encode(bufB, width, rows: 2);

            Assert.Equal(bufA[(width)..], bufB[(width)..]);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. Special value tests
    // ═══════════════════════════════════════════════════════════════════════════

    public class ByteSpecialValueTests
    {
        [Fact]
        public void AllZeros_RoundTrip()
        {
            var buf = new byte[64];
            BytePredictor.Encode(buf, width: 64, rows: 1);
            BytePredictor.Decode(buf, width: 64, rows: 1);
            Assert.All(buf, b => Assert.Equal(0, b));
        }

        [Fact]
        public void AllZeros_EncodeIsAllZeros()
        {
            var buf = new byte[32];
            BytePredictor.Encode(buf, width: 32, rows: 1);
            Assert.All(buf, b => Assert.Equal(0, b));
        }

        [Fact]
        public void All0xFF_WrappingRoundTrip()
        {
            // Each byte is 0xFF; deltas are zero (or wrap to zero)
            var buf = new byte[64];
            Array.Fill(buf, (byte)0xFF);
            var original = (byte[])buf.Clone();

            BytePredictor.Encode(buf, width: 64, rows: 1);
            BytePredictor.Decode(buf, width: 64, rows: 1);

            Assert.Equal(original, buf);
        }

        [Fact]
        public void Alternating0x00_0xFF_RoundTrip()
        {
            var buf = new byte[64];
            for (int i = 0; i < 64; i++)
                buf[i] = (byte)((i & 1) == 0 ? 0x00 : 0xFF);
            var original = (byte[])buf.Clone();

            BytePredictor.Encode(buf, width: 64, rows: 1);
            BytePredictor.Decode(buf, width: 64, rows: 1);

            Assert.Equal(original, buf);
        }

        [Fact]
        public void AscendingSequence_RoundTrip()
        {
            var buf = new byte[256];
            for (int i = 0; i < 256; i++)
                buf[i] = (byte)i;
            var original = (byte[])buf.Clone();

            BytePredictor.Encode(buf, width: 256, rows: 1);
            BytePredictor.Decode(buf, width: 256, rows: 1);

            Assert.Equal(original, buf);
        }

        [Fact]
        public void DescendingSequence_RoundTrip()
        {
            var buf = new byte[256];
            for (int i = 0; i < 256; i++)
                buf[i] = (byte)(255 - i);
            var original = (byte[])buf.Clone();

            BytePredictor.Encode(buf, width: 256, rows: 1);
            BytePredictor.Decode(buf, width: 256, rows: 1);

            Assert.Equal(original, buf);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 5. Scalar-path correctness (widths 1-16 exercise scalar paths)
    // ═══════════════════════════════════════════════════════════════════════════

    public class ByteScalarCorrectnessTests
    {
        [Fact]
        public void Width1_EncodeDecodeRoundTrip()
        {
            var buf = new byte[] { 0x7F };
            BytePredictor.Encode(buf, width: 1, rows: 1);
            BytePredictor.Decode(buf, width: 1, rows: 1);
            Assert.Equal(0x7F, buf[0]);
        }

        [Fact]
        public void Width2_EncodeDecodeRoundTrip()
        {
            var buf = new byte[] { 0x10, 0x20 };
            var orig = (byte[])buf.Clone();
            BytePredictor.Encode(buf, width: 2, rows: 1);
            BytePredictor.Decode(buf, width: 2, rows: 1);
            Assert.Equal(orig, buf);
        }

        [Fact]
        public void Width3_EncodeDecodeRoundTrip()
        {
            var buf = new byte[] { 0x01, 0x02, 0x03 };
            var orig = (byte[])buf.Clone();
            BytePredictor.Encode(buf, width: 3, rows: 1);
            BytePredictor.Decode(buf, width: 3, rows: 1);
            Assert.Equal(orig, buf);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(4)]
        [InlineData(7)]
        [InlineData(8)]
        [InlineData(15)]
        [InlineData(16)]
        public void ScalarWidth_EncodeMatchesReference(int width)
        {
            byte[] row = ByteTestHelpers.RandomBytes(width, seed: width * 99);
            var expected = ByteTestHelpers.ReferenceEncodeStride1(row);
            var actual = (byte[])row.Clone();
            BytePredictor.Encode(actual, width, rows: 1);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Width1_MultiRow_EachRowIndependent()
        {
            byte[] data = { 0x01, 0x02, 0x03, 0x04 };
            var buf = (byte[])data.Clone();
            BytePredictor.Encode(buf, width: 1, rows: 4);
            // width=1: no delta — each byte is its own row and stays unchanged
            Assert.Equal(data, buf);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 6. Stride tests (bytesPerSample > 1)
    // ═══════════════════════════════════════════════════════════════════════════

    public class ByteStrideTests
    {
        // ── bps=2 (16-bit samples) ─────────────────────────────────────────────

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(4)]
        [InlineData(8)]
        [InlineData(16)]
        [InlineData(32)]
        [InlineData(65)]
        public void Bps2_SingleRow_RoundTrip(int width)
        {
            int bps = 2;
            var original = ByteTestHelpers.RandomBytes(width * bps, seed: width);
            var buf = (byte[])original.Clone();

            BytePredictor.Encode(buf, width, rows: 1, bytesPerSample: bps);
            BytePredictor.Decode(buf, width, rows: 1, bytesPerSample: bps);

            Assert.Equal(original, buf);
        }

        [Theory]
        [InlineData(4, 3)]
        [InlineData(16, 4)]
        [InlineData(32, 8)]
        public void Bps2_MultiRow_RoundTrip(int width, int rows)
        {
            int bps = 2;
            var original = ByteTestHelpers.RandomBytes(width * rows * bps, seed: width ^ rows);
            var buf = (byte[])original.Clone();

            BytePredictor.Encode(buf, width, rows, bytesPerSample: bps);
            BytePredictor.Decode(buf, width, rows, bytesPerSample: bps);

            Assert.Equal(original, buf);
        }

        [Fact]
        public void Bps2_EncodeMatchesReference()
        {
            int width = 8, bps = 2;
            byte[] row = ByteTestHelpers.RandomBytes(width * bps, seed: 42);
            var expected = ByteTestHelpers.ReferenceEncodeStrided(row, width, bps);
            var actual = (byte[])row.Clone();
            BytePredictor.Encode(actual, width, rows: 1, bytesPerSample: bps);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Bps2_DecodeMatchesReference()
        {
            int width = 8, bps = 2;
            byte[] original = ByteTestHelpers.RandomBytes(width * bps, seed: 77);
            var encoded = ByteTestHelpers.ReferenceEncodeStrided(original, width, bps);
            var expected = ByteTestHelpers.ReferenceDecodeStrided(encoded, width, bps);
            var actual = (byte[])encoded.Clone();
            BytePredictor.Decode(actual, width, rows: 1, bytesPerSample: bps);
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void Bps2_FirstSampleNeverDeltaEncoded()
        {
            int width = 4, bps = 2;
            byte[] original = ByteTestHelpers.RandomBytes(width * bps, seed: 13);
            var buf = (byte[])original.Clone();
            BytePredictor.Encode(buf, width, rows: 1, bytesPerSample: bps);
            Assert.Equal(original[0], buf[0]);
            Assert.Equal(original[1], buf[1]);
        }

        // ── bps=3 (24-bit samples) ─────────────────────────────────────────────

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(4)]
        [InlineData(8)]
        [InlineData(16)]
        public void Bps3_SingleRow_RoundTrip(int width)
        {
            int bps = 3;
            var original = ByteTestHelpers.RandomBytes(width * bps, seed: width);
            var buf = (byte[])original.Clone();
            BytePredictor.Encode(buf, width, rows: 1, bytesPerSample: bps);
            BytePredictor.Decode(buf, width, rows: 1, bytesPerSample: bps);
            Assert.Equal(original, buf);
        }

        [Fact]
        public void Bps3_EncodeMatchesReference()
        {
            int width = 5, bps = 3;
            byte[] row = ByteTestHelpers.RandomBytes(width * bps, seed: 99);
            var expected = ByteTestHelpers.ReferenceEncodeStrided(row, width, bps);
            var actual = (byte[])row.Clone();
            BytePredictor.Encode(actual, width, rows: 1, bytesPerSample: bps);
            Assert.Equal(expected, actual);
        }

        // ── bps=4 (32-bit samples) ─────────────────────────────────────────────

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(4)]
        [InlineData(8)]
        [InlineData(16)]
        [InlineData(32)]
        public void Bps4_SingleRow_RoundTrip(int width)
        {
            int bps = 4;
            var original = ByteTestHelpers.RandomBytes(width * bps, seed: width);
            var buf = (byte[])original.Clone();
            BytePredictor.Encode(buf, width, rows: 1, bytesPerSample: bps);
            BytePredictor.Decode(buf, width, rows: 1, bytesPerSample: bps);
            Assert.Equal(original, buf);
        }

        [Fact]
        public void Bps4_EncodeMatchesReference()
        {
            int width = 8, bps = 4;
            byte[] row = ByteTestHelpers.RandomBytes(width * bps, seed: 123);
            var expected = ByteTestHelpers.ReferenceEncodeStrided(row, width, bps);
            var actual = (byte[])row.Clone();
            BytePredictor.Encode(actual, width, rows: 1, bytesPerSample: bps);
            Assert.Equal(expected, actual);
        }

        // ── bps=8 (64-bit samples) ─────────────────────────────────────────────

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(4)]
        [InlineData(8)]
        public void Bps8_SingleRow_RoundTrip(int width)
        {
            int bps = 8;
            var original = ByteTestHelpers.RandomBytes(width * bps, seed: width);
            var buf = (byte[])original.Clone();
            BytePredictor.Encode(buf, width, rows: 1, bytesPerSample: bps);
            BytePredictor.Decode(buf, width, rows: 1, bytesPerSample: bps);
            Assert.Equal(original, buf);
        }

        [Fact]
        public void Bps8_EncodeMatchesReference()
        {
            int width = 4, bps = 8;
            byte[] row = ByteTestHelpers.RandomBytes(width * bps, seed: 456);
            var expected = ByteTestHelpers.ReferenceEncodeStrided(row, width, bps);
            var actual = (byte[])row.Clone();
            BytePredictor.Encode(actual, width, rows: 1, bytesPerSample: bps);
            Assert.Equal(expected, actual);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 7. API contract tests
    // ═══════════════════════════════════════════════════════════════════════════

    public class ByteApiContractTests
    {
        [Fact]
        public void Encode_EmptyTile_DoesNotThrow()
        {
            var buf = Array.Empty<byte>();
            BytePredictor.Encode(buf, width: 0, rows: 0);
        }

        [Fact]
        public void Decode_EmptyTile_DoesNotThrow()
        {
            var buf = Array.Empty<byte>();
            BytePredictor.Decode(buf, width: 0, rows: 0);
        }

        [Fact]
        public void Encode_IsInPlace_ReturnsVoid()
        {
            var buf = new byte[] { 0x01, 0x02, 0x03, 0x04 };
            var orig = (byte[])buf.Clone();
            BytePredictor.Encode(buf, width: 4, rows: 1);
            Assert.NotEqual(orig, buf);
        }

        [Fact]
        public void Decode_IsInPlace_ReturnsVoid()
        {
            var buf = new byte[] { 0x01, 0x02, 0x03, 0x04 };
            BytePredictor.Encode(buf, width: 4, rows: 1);
            var encoded = (byte[])buf.Clone();
            BytePredictor.Decode(buf, width: 4, rows: 1);
            Assert.NotEqual(encoded, buf);
        }

        [Fact]
        public void EncodeIsIdempotentAfterDecodeEncode()
        {
            byte[] data = ByteTestHelpers.RandomBytes(64, seed: 1234);
            var buf = (byte[])data.Clone();

            BytePredictor.Encode(buf, width: 64, rows: 1);
            var firstEncode = (byte[])buf.Clone();

            BytePredictor.Decode(buf, width: 64, rows: 1);
            BytePredictor.Encode(buf, width: 64, rows: 1);

            Assert.Equal(firstEncode, buf);
        }

        [Fact]
        public void Encode_Width1_IsNoOp()
        {
            var buf = new byte[] { 0xAA };
            BytePredictor.Encode(buf, width: 1, rows: 1);
            Assert.Equal(0xAA, buf[0]);
        }

        [Fact]
        public void Decode_Width1_IsNoOp()
        {
            var buf = new byte[] { 0xBB };
            BytePredictor.Encode(buf, width: 1, rows: 1); // encode does nothing for width=1
            BytePredictor.Decode(buf, width: 1, rows: 1);
            Assert.Equal(0xBB, buf[0]);
        }
    }
}
