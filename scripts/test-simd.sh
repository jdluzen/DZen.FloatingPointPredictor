#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
configuration="${CONFIGURATION:-Release}"
target_framework="${TARGET_FRAMEWORK:-net10.0}"
solution="$repo_root/DZen.FloatingPointPredictor.slnx"
test_output="$repo_root/DZen.FloatingPointPredictor.Tests/bin/$configuration/$target_framework"
test_assembly="DZen.FloatingPointPredictor.Tests.dll"

echo "Building $configuration for $target_framework..."
dotnet build "$solution" \
  --configuration "$configuration" \
  --maxcpucount:1 \
  --disable-build-servers

if [[ ! -f "$test_output/$test_assembly" ]]; then
  echo "Test assembly not found: $test_output/$test_assembly" >&2
  exit 1
fi

test_dir="$(mktemp -d "${TMPDIR:-/tmp}/dzen-fpp-simd.XXXXXX")"
cleanup() {
  rm -rf -- "$test_dir"
}
trap cleanup EXIT

# Some mounted filesystems cannot mmap .NET runtimeconfig files. Running the
# already-built test host from a local temporary directory avoids that issue and
# also means every profile executes the exact same binaries.
cp -R "$test_output/." "$test_dir/"

profile_count=0
run_profile() {
  local name="$1"
  shift

  echo
  echo "=== SIMD profile: $name ==="
  env "$@" dotnet vstest "$test_dir/$test_assembly" \
    --Logger:"console;verbosity=minimal"
  profile_count=$((profile_count + 1))
}

cpu_features=""
case "$(uname -s)" in
  Linux)
    if command -v lscpu >/dev/null 2>&1; then
      cpu_features="$(lscpu 2>/dev/null | tr '[:upper:]' '[:lower:]')"
    elif [[ -r /proc/cpuinfo ]]; then
      cpu_features="$(tr '[:upper:]' '[:lower:]' </proc/cpuinfo)"
    fi
    ;;
  Darwin)
    cpu_features="$(sysctl -a 2>/dev/null | tr '[:upper:]' '[:lower:]')"
    ;;
esac

has_cpu_feature() {
  local feature="$1"
  grep -Eq "(^|[^[:alnum:]_])${feature}([^[:alnum:]_]|$)" <<<"$cpu_features"
}

architecture="$(uname -m | tr '[:upper:]' '[:lower:]')"

# The native run exercises the highest paths exposed by the current runtime.
# On AVX-512 hosts this is the AVX-512 VBMI/BW run.
run_profile "native automatic dispatch"

case "$architecture" in
  x86_64|amd64)
    if has_cpu_feature avx2; then
      run_profile "AVX2" \
        DOTNET_EnableAVX512=0
    else
      echo "Skipping AVX2: not exposed by this host."
    fi

    if has_cpu_feature ssse3 && has_cpu_feature sse2; then
      run_profile "SSSE3 for Fp32 / SSE2 for Byte" \
        DOTNET_EnableAVX512=0 \
        DOTNET_EnableAVX2=0
    else
      echo "Skipping SSSE3/SSE2: not exposed by this host."
    fi

    if has_cpu_feature sse2; then
      run_profile "SSE2 for Byte / scalar Fp32" \
        DOTNET_EnableAVX512=0 \
        DOTNET_EnableAVX2=0 \
        DOTNET_EnableSSE42=0
    else
      echo "Skipping SSE2: not exposed by this host."
    fi
    ;;
  aarch64|arm64)
    # AdvSimd is part of the ARM64 platform. The native profile above exercises
    # the NEON paths; the scalar profile below provides the fallback coverage.
    echo "Native ARM64 profile exercises AdvSimd/NEON."
    ;;
  *)
    echo "No additional forced SIMD profiles are defined for $architecture."
    ;;
esac

run_profile "scalar" \
  DOTNET_EnableHWIntrinsic=0

echo
echo "All $profile_count available SIMD/scalar profiles passed."
