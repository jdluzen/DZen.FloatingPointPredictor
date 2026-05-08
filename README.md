# DZen.FloatingPointPredictor

Hardware-accelerated implementations of **TIFF Predictor = 2** (horizontal differencing) and **TIFF Predictor = 3** (floating-point predictor, TIFF Technical Note 3) for .NET. Both predictors use the same algorithms as GDAL, libtiff, and the Cloud-Optimized GeoTIFF (COG) / ZSTD pipeline.

[![NuGet](https://img.shields.io/nuget/v/DZen.FloatingPointPredictor)](https://www.nuget.org/packages/DZen.FloatingPointPredictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-blue)](https://dotnet.microsoft.com)

---

## What it does

### Predictor = 3 — Floating-Point Predictor

Float32 raster data compresses poorly because entropy coders (ZSTD, LZW, Deflate) don't exploit the byte-level structure of IEEE 754 values. The floating-point predictor reorganises the raw bytes of each tile row before compression, dramatically increasing the compressibility of elevation, satellite imagery, and other continuous-field rasters.

The transform has two steps applied per row:

**1. Byte-plane shuffle** — the four raw bytes of every `float` are separated into four contiguous planes ordered MSB-first:

```
Input (interleaved):   [B3 B2 B1 B0 | B3 B2 B1 B0 | ...]   ← N floats
Output (planar):       [B3 B3 … B3 | B2 B2 … B2 | B1 B1 … | B0 B0 …]
```

**2. Horizontal delta encoding** — each byte in every plane is replaced by `current − previous`. This concentrates entropy in the sign and exponent bytes, which change slowly across spatially coherent data.

Decoding applies the exact inverse: cumulative sum to undo the delta, then unshuffle the bytes back to interleaved float32 layout.

### Predictor = 2 — Horizontal Differencing

A general-purpose byte-level differencing predictor. Each sample in a row is replaced by its difference from the preceding sample, with modular byte arithmetic:

```
s[0]   = unchanged
s[i]   = s[i] − s[i−1]   (encode, right-to-left)
s[i]   = s[i] + s[i−1]   (decode, left-to-right)
```

Supports configurable sample widths via the `bytesPerSample` parameter (1-byte, 2-byte, 3-byte, 4-byte, 8-byte). When `bytesPerSample == 1` the inner loop maps directly to SIMD byte subtraction. For multi-byte samples a scalar strided path is used.

---

## Hardware dispatch

Both predictors use runtime dispatch via `IsSupported` checks resolved at JIT time — there is no per-call overhead. All paths produce bit-identical output.

### Fp32Predictor (Predictor = 3)

The byte-shuffle step is the inner-loop bottleneck:

| ISA | Instruction | Floats / iteration | Bytes / iteration |
|-----|-------------|-------------------|-------------------|
| **AVX-512 VBMI** | `VPERMB` | 16 | 64 |
| **AVX2** | `VPSHUFB` (×2 lanes) | 8 | 32 |
| **SSSE3** | `PSHUFB` | 4 | 16 |
| **ARM NEON** | `AdvSimd.UnzipEven/Odd` | 4 | 16 |
| **Scalar** | Plain C# (portable fallback) | 1 | 4 |

### BytePredictor (Predictor = 2)

When `bytesPerSample == 1`, the delta step maps naturally to SIMD byte subtraction:

| ISA | Instruction | Bytes / iteration |
|-----|-------------|-------------------|
| **AVX-512BW** | `VPSUBB` | 64 |
| **AVX2** | `VPSUBB` | 32 |
| **SSE2** | `PSUBB` | 16 |
| **ARM NEON** | `AdvSimd.Subtract` | 16 |
| **Scalar** | Plain C# (portable fallback) | 1 |

For `bytesPerSample > 1` only the scalar path is used.

---

## Installation

```
dotnet add package DZen.FloatingPointPredictor
```

Or via the NuGet Package Manager:

```
Install-Package DZen.FloatingPointPredictor
```

Requires **.NET 10.0** or later. The package targets `net10.0`. No unsafe code — all SIMD access uses `Vector.LoadUnsafe`/`Vector.StoreUnsafe` with span-based refs.

---

## API

```csharp
namespace DZen.FloatingPointPredictor;

// ── Floating-Point Predictor (TIFF Predictor = 3) ────────────────────────

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

// ── Horizontal Differencing Predictor (TIFF Predictor = 2) ───────────────

public static class BytePredictor
{
    /// <summary>
    /// Applies horizontal differencing encode in-place.
    /// The buffer is treated as rows × width samples of bytesPerSample bytes each.
    /// </summary>
    public static void Encode(Span<byte> tile, int width, int rows,
                              int bytesPerSample = 1);

    /// <summary>
    /// Reverses the horizontal differencing transform in-place.
    /// </summary>
    public static void Decode(Span<byte> tile, int width, int rows,
                              int bytesPerSample = 1);
}
```

All methods operate **in-place** on the caller's buffer. All classes are **stateless and thread-safe**.

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

When writing a Cloud-Optimized GeoTIFF with Predictor = 3, apply `Fp32Predictor.Encode` to each tile buffer immediately before passing it to the TIFF tile write call. The TIFF tag signals to readers (GDAL, libtiff, etc.) that they must call the inverse transform after decompression.

### BytePredictor usage (Predictor = 2)

```csharp
using DZen.FloatingPointPredictor;

byte[] tile = ReadTileFromRaster(...);
int width = 512, rows = 512;

// bytesPerSample defaults to 1 (byte-level delta)
BytePredictor.Encode(tile, width, rows);

// For 16-bit samples, specify bytesPerSample = 2
BytePredictor.Encode(tile, width, rows, bytesPerSample: 2);

// Decoding reverses the transform
BytePredictor.Decode(tile, width, rows);

// For multi-byte samples, match the same bytesPerSample
BytePredictor.Decode(tile, width, rows, bytesPerSample: 2);
```

---

## Algorithm detail

### Predictor = 3 (Fp32Predictor)

#### Byte index mapping

For float `i` (0-based) in a row of `width` floats, the shuffle places:

| Plane | Output index | Source byte | Description |
|-------|-------------|-------------|-------------|
| 0 | `0·width + i` | `input[4i+3]` | MSB (sign + exponent high) |
| 1 | `1·width + i` | `input[4i+2]` | Exponent low + mantissa high |
| 2 | `2·width + i` | `input[4i+1]` | Mantissa mid |
| 3 | `3·width + i` | `input[4i+0]` | LSB (mantissa low) |

#### Delta encoding

After the shuffle, for each of the four planes independently:

```
encoded[plane][0]   = shuffled[plane][0]               ← first sample unchanged
encoded[plane][i]   = shuffled[plane][i] - shuffled[plane][i-1]  for i ≥ 1
```

Delta arithmetic is modular byte arithmetic (wraps naturally on overflow), which makes encoding and decoding exact inverses with no range clamping.

#### Why MSB first?

The sign bit and most of the exponent live in byte 3 (MSB). For spatially coherent fields such as elevation or reflectance, adjacent pixels share similar exponents. Placing the MSB in plane 0 gives the entropy coder the most compressible bytes first and allows early termination in some codecs.

### Predictor = 2 (BytePredictor)

For `bytesPerSample = B`, the row of `width` samples is processed:

```
encode:  s[i] = s[i] − s[i−1]      (i from width−1 down to 1, right-to-left)
decode:  s[i] = s[i] + s[i−1]      (i from 1 to width−1, left-to-right)
```

Each sample is `B` bytes; subtraction is done byte-by-byte with modular arithmetic. The first sample of every row carries the raw value and is never differenced. For `bytesPerSample == 1`, right-to-left overlapping SIMD loads and left-to-right parallel prefix-sum kernels (shift-left + add, log₂(N) steps) accelerate the inner loop.

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

The test suite is in `DZen.FloatingPointPredictor.Tests` and uses **xUnit v3**. Tests are organised into thirteen classes covering both predictors — together they verify correctness at every SIMD width, every SIMD path boundary, all IEEE 754 special values (Predictor 3) and byte wrapping edge cases (Predictor 2), and against independent reference implementations.

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

#### Fp32Predictor (Predictor = 3)

**1. `RoundTripTests`** — Encode→Decode recovers original bytes exactly. Parameterised over widths 1–512 covering every SIMD chunk boundary (scalar, SSSE3, AVX2, AVX-512).

**2. `ConformanceTests`** — Byte-level assertions against TIFF TN3 spec and independent reference encoder.

**3. `RowIndependenceTests`** — Each row is processed independently; delta does not cross row boundaries.

**4. `SpecialValueTests`** — NaN, ±Inf, ±0, subnormals, MaxValue survive round-trip without bit corruption.

**5. `ScalarCorrectnessTests`** — Widths 1–3 exercise only the scalar path (no SIMD fires below width 4).

**6. `ApiContractTests`** — Empty input no-throw, in-place mutation, idempotence of Encode(Decode(Encode(x))).

#### BytePredictor (Predictor = 2)

**7. `ByteRoundTripTests`** — Round-trip across all SIMD boundaries (1–512 bytes per row).

**8. `ByteConformanceTests`** — Byte-level correctness including modular wrap (e.g. `0x10 − 0x80 = 0x90`).

**9. `ByteRowIndependenceTests`** — Per-row independence verification.

**10. `ByteSpecialValueTests`** — Zeros, 0xFF wrapping, ascending/descending/alternating byte sequences.

**11. `ByteScalarCorrectnessTests`** — Small widths (1–16) forcing scalar fallback paths.

**12. `ByteStrideTests`** — Multi-byte samples: `bytesPerSample` ∈ {2, 3, 4, 8} with matching reference.

**13. `ByteApiContractTests`** — In-place semantics, empty input, idempotence, width-1 no-op.

---

## Platform compatibility

Both predictors dispatch to the fastest SIMD path available at runtime:

| Platform | Architecture | SIMD path used |
|----------|-------------|----------------|
| Windows / Linux / macOS | x64 with AVX-512 VBMI | AVX-512 (16 floats/iter) |
| Windows / Linux / macOS | x64 with AVX2 | AVX2 (8 floats/iter) |
| Windows / Linux / macOS | x64 with SSSE3 (any Sandy Bridge+) | SSSE3 (4 floats/iter) |
| Linux / macOS | ARM64 (Apple M-series, Ampere, Graviton) | NEON (4 floats/iter) |
| Any | Any (including x86, WASM) | Scalar |

The JIT resolves `IsSupported` checks at compile time for the current process's CPU; there is no runtime branching inside the hot loop. No `unsafe` code or `AllowUnsafeBlocks` is required — all SIMD access is through `Vector.LoadUnsafe`/`Vector.StoreUnsafe` with span-based refs.

---

## License

MIT — see [LICENSE](LICENSE).

---

## References

- [TIFF Technical Note 3 — Floating-Point Predictor](https://download.osgeo.org/geotiff/specs/TIFF_Floating-Point_Predictor_Support.pdf)
- [Cloud-Optimized GeoTIFF specification](https://cogeo.org)
- [GDAL floating-point predictor implementation](https://github.com/OSGeo/gdal)
