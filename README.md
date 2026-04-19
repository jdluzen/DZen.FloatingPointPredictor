# DZen.FloatingPointPredictor

A hardware-accelerated implementation of the **TIFF Floating-Point Predictor** (TIFF Technical Note 3, Predictor = 3) for .NET. This is the same algorithm used by GDAL, libtiff, and the Cloud-Optimized GeoTIFF (COG) / ZSTD pipeline for float32 raster data.

[![NuGet](https://img.shields.io/nuget/v/DZen.FloatingPointPredictor)](https://www.nuget.org/packages/DZen.FloatingPointPredictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-blue)](https://dotnet.microsoft.com)

---

## What it does

Float32 raster data compresses poorly because entropy coders (ZSTD, LZW, Deflate) don't exploit the byte-level structure of IEEE 754 values. The floating-point predictor reorganises the raw bytes of each tile row before compression, dramatically increasing the compressibility of elevation, satellite imagery, and other continuous-field rasters.

The transform has two steps applied per row:

**1. Byte-plane shuffle** — the four raw bytes of every `float` are separated into four contiguous planes ordered MSB-first:

```
Input (interleaved):   [B3 B2 B1 B0 | B3 B2 B1 B0 | ...]   ← N floats
Output (planar):       [B3 B3 … B3 | B2 B2 … B2 | B1 B1 … | B0 B0 …]
```

**2. Horizontal delta encoding** — each byte in every plane is replaced by `current − previous`. This concentrates entropy in the sign and exponent bytes, which change slowly across spatially coherent data.

Decoding applies the exact inverse: cumulative sum to undo the delta, then unshuffle the bytes back to interleaved float32 layout.

---

## Hardware dispatch

The byte-shuffle step is the inner-loop bottleneck. At startup the library selects the fastest available path with no configuration required:

| ISA | Instruction | Floats / iteration | Bytes / iteration |
|-----|-------------|-------------------|-------------------|
| **AVX-512 VBMI** | `VPERMB` | 16 | 64 |
| **AVX2** | `VPSHUFB` (×2 lanes) | 8 | 32 |
| **SSSE3** | `PSHUFB` | 4 | 16 |
| **ARM NEON** | `AdvSimd.UnzipEven/Odd` | 4 | 16 |
| **Scalar** | Plain C# (portable fallback) | 1 | 4 |

The dispatch is a chain of `IsSupported` checks resolved once by the JIT — there is no per-call overhead, and no configuration file or environment variable is needed. All paths produce bit-identical output.

---

## Installation

```
dotnet add package DZen.FloatingPointPredictor
```

Or via the NuGet Package Manager:

```
Install-Package DZen.FloatingPointPredictor
```

Requires **.NET 10.0** or later. The package targets `net10.0` and uses `AllowUnsafeBlocks` for the raw pointer paths inside the SSSE3, AVX2, AVX-512, and NEON implementations.

---

## API

The entire public surface is a single static class with two methods:

```csharp
namespace DZen.FloatingPointPredictor;

public static class Fp32Predictor
{
    /// <summary>
    /// Applies the floating-point predictor encode transform in-place.
    /// The buffer is treated as rows × width float32 values in row-major order.
    /// Each row is processed independently (as per TIFF TN3).
    /// </summary>
    public static void Encode(Span<byte> tile, int width, int rows);

    /// <summary>
    /// Reverses the floating-point predictor transform in-place.
    /// </summary>
    public static void Decode(Span<byte> tile, int width, int rows);
}
```

Both methods operate **in-place** on the caller's buffer. There are no allocations on the hot path (the scalar fallback allocates a small temporary; all SIMD paths avoid it). Both methods are **stateless and thread-safe**.

---

## Usage

### Encoding before compression

```csharp
using DZen.FloatingPointPredictor;

// tile is a raw byte buffer: rows × width × sizeof(float) bytes
byte[] tile = ReadTileFromRaster(...);   // float32, row-major
int width = 512, rows = 512;

Fp32Predictor.Encode(tile, width, rows);

// hand tile to your compressor (ZSTD, Deflate, LZW…)
byte[] compressed = Zstd.Compress(tile);
```

### Decoding after decompression

```csharp
byte[] compressed = ReadCompressedTileFromFile(...);
byte[] tile = Zstd.Decompress(compressed);

Fp32Predictor.Decode(tile, width, rows);

// reinterpret as float32
Span<float> floats = MemoryMarshal.Cast<byte, float>(tile);
```

### Integrating with a TIFF writer

When writing a Cloud-Optimized GeoTIFF with Predictor = 3, apply `Encode` to each tile buffer immediately before passing it to the TIFF tile write call. The TIFF tag signals to readers (GDAL, libtiff, etc.) that they must call the inverse transform after decompression.

---

## Algorithm detail

### Byte index mapping

For float `i` (0-based) in a row of `width` floats, the shuffle places:

| Plane | Output index | Source byte | Description |
|-------|-------------|-------------|-------------|
| 0 | `0·width + i` | `input[4i+3]` | MSB (sign + exponent high) |
| 1 | `1·width + i` | `input[4i+2]` | Exponent low + mantissa high |
| 2 | `2·width + i` | `input[4i+1]` | Mantissa mid |
| 3 | `3·width + i` | `input[4i+0]` | LSB (mantissa low) |

### Delta encoding

After the shuffle, for each of the four planes independently:

```
encoded[plane][0]   = shuffled[plane][0]               ← first sample unchanged
encoded[plane][i]   = shuffled[plane][i] - shuffled[plane][i-1]  for i ≥ 1
```

Delta arithmetic is modular byte arithmetic (wraps naturally on overflow), which makes encoding and decoding exact inverses with no range clamping.

### Why MSB first?

The sign bit and most of the exponent live in byte 3 (MSB). For spatially coherent fields such as elevation or reflectance, adjacent pixels share similar exponents. Placing the MSB in plane 0 gives the entropy coder the most compressible bytes first and allows early termination in some codecs.

---

## Building from source

```bash
git clone https://github.com/jdluzen/DZen.FloatingPointPredictor
cd DZen.FloatingPointPredictor
dotnet build -c Release
```

To run the tests:

```bash
dotnet test DZen.FloatingPointPredictor.Tests
```

---

## Tests

The test suite is in `DZen.FloatingPointPredictor.Tests` and uses **xUnit**. Tests are organised into six classes that together verify correctness at every SIMD width, at every SIMD path boundary, for all IEEE 754 special values, and against a reference implementation independent of the production code.

### Running

```bash
dotnet test DZen.FloatingPointPredictor.Tests
# or with verbose output
dotnet test DZen.FloatingPointPredictor.Tests --logger "console;verbosity=detailed"
```

Code coverage is collected via **Coverlet** (included as a test dependency) if your CI runner supports it:

```bash
dotnet test DZen.FloatingPointPredictor.Tests --collect:"XPlat Code Coverage"
```

---

### Test classes

#### 1. `RoundTripTests`

Verifies that `Encode` followed immediately by `Decode` recovers the original bytes **exactly** (bit-for-bit). Tests cover every interesting width boundary:

- `SingleRow_RoundTrip` — parameterised over widths 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128, 256, 512. These boundary values are chosen to exercise the scalar-only tail (1–3), the first SSSE3 chunk (4), the boundary between SSSE3 and AVX2 (7–9), the AVX2 boundary (15–17), and the AVX-512 boundary (31–33).
- `MultiRow_RoundTrip` — parameterised over width/row combinations (1×1 up to 512×512) to confirm that multi-row tiles are handled correctly.
- `StandardCogTile_512x512_RoundTrip` — a full 512 × 512 tile, the most common COG tile size in production.

#### 2. `ConformanceTests`

Validates exact byte-level output against the TIFF TN3 specification and against an independent reference implementation (`TestHelpers.ReferenceEncode`) written without any SIMD.

- `Encode_SingleFloat_BytePlanesAreMsbFirst` — asserts that `1.0f` (IEEE 754: `0x3F800000`) encodes to `[0x3F, 0x80, 0x00, 0x00]`, confirming the MSB-first plane order.
- `Encode_TwoFloats_DeltaEncodesEachPlane` — checks a hard-coded expected result for `[1.0f, 2.0f]`, including the modular wrap in plane 1 (`0x00 − 0x80 = 0x80`).
- `Encode_FirstSampleOfEachPlane_IsNeverDeltaEncoded` — confirms the first byte of each plane carries the raw shuffled byte and is not differenced.
- `Encode_MatchesReferenceImplementation` — parameterised over widths 1–64 (covering scalar tail and each SIMD tier) comparing byte-for-byte against the reference.
- `Decode_IsExactInverseOfEncode` — confirms the reference encoder and decoder cancel exactly.

#### 3. `RowIndependenceTests`

TIFF TN3 requires that each row is processed **independently** — the delta must not carry state across a row boundary.

- `MultiRowEncode_EqualsIndividualRowEncodes` — encodes a multi-row tile in one call and compares it against encoding each row separately. Parameterised over several width/row combinations.
- `DeltaDoesNotCarryAcrossRowBoundary` — builds two tiles with the same row 1 but different row 0, encodes both, and asserts that row 1's encoded bytes are identical in both tiles regardless of what preceded them.

#### 4. `SpecialValueTests`

Ensures that IEEE 754 edge cases survive the round-trip without any bit corruption. NaN comparison uses raw bit equality (`BitConverter.SingleToUInt32Bits`) to avoid the IEEE rule that NaN ≠ NaN.

- `AllZeros_RoundTrip` and `AllZeros_EncodeIsAllZeros` — `0.0f` is `0x00000000`; after shuffle and delta the result must also be all zeros.
- `SpecialValues_RoundTrip` — parameterised over `NaN`, `+Inf`, `-Inf`, `float.MaxValue`, `float.MinValue`, `float.Epsilon`, `+0.0f`, `-0.0f`.
- `SpecialValues_EncodeMatchesReference` — the same special values compared byte-for-byte against the reference encoder.
- `MixedSpecialAndNormalValues_RoundTrip` — a 16-element row that mixes all of the above with normal floats, covering a full AVX2 iteration.

#### 5. `ScalarCorrectnessTests`

No SIMD path fires for widths 1, 2, or 3 (all paths require at least 4 floats). These tests exercise the scalar fallback in isolation:

- `Width1_EncodeDecodeRoundTrip`, `Width2_EncodeDecodeRoundTrip`, `Width3_EncodeDecodeRoundTrip` — basic round-trip for each scalar-only width.
- `ScalarWidth_EncodeMatchesReference` — parameterised over widths 1–3, comparing against the reference encoder.
- `Width1_MultipleRows_EachRowIsIndependent` — with width = 1 there is no delta (single sample per row), so the encode is a pure byte shuffle; asserts that each of four rows encodes to exactly the shuffled bytes of its float.

#### 6. `ApiContractTests`

Documents and verifies the observable behaviour of the public API surface:

- `Encode_EmptyTile_DoesNotThrow` / `Decode_EmptyTile_DoesNotThrow` — zero-width, zero-row input must not throw.
- `Encode_IsInPlace_ReturnsVoid` / `Decode_IsInPlace_ReturnsVoid` — both methods modify the caller's span in-place and return `void`.
- `EncodeIsIdempotentAfterDecodeEncode` — verifies that `Encode(Decode(Encode(x))) == Encode(x)`, a useful property for codec pipeline correctness.

---

## Platform compatibility

| Platform | Architecture | SIMD path used |
|----------|-------------|----------------|
| Windows / Linux / macOS | x64 with AVX-512 VBMI | AVX-512 (16 floats/iter) |
| Windows / Linux / macOS | x64 with AVX2 | AVX2 (8 floats/iter) |
| Windows / Linux / macOS | x64 with SSSE3 (any Sandy Bridge+) | SSSE3 (4 floats/iter) |
| Linux / macOS | ARM64 (Apple M-series, Ampere, Graviton) | NEON (4 floats/iter) |
| Any | Any (including x86, WASM) | Scalar |

The JIT resolves `IsSupported` checks at compile time for the current process's CPU; there is no runtime branching inside the hot loop.

---

## License

MIT — see [LICENSE](LICENSE).

---

## References

- [TIFF Technical Note 3 — Floating-Point Predictor](https://download.osgeo.org/geotiff/specs/TIFF_Floating-Point_Predictor_Support.pdf)
- [Cloud-Optimized GeoTIFF specification](https://cogeo.org)
- [GDAL floating-point predictor implementation](https://github.com/OSGeo/gdal)
