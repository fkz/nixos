# nix build .#avalon
# For AMD GPUs, override rocmSupport and rocmGpuTargets for the target hardware.
{ lib, stdenv, fetchFromGitHub, cmake, ninja, pkg-config, openssl
, rocmSupport ? false, rocmPackages, rocmGpuTargets ? [ "gfx1150" ]
}:

stdenv.mkDerivation {
  pname = "avalon";
  version = "0-unstable-2026-11-05";

  src = fetchFromGitHub {
    owner = "freriadoc";
    repo = "Avalon";
    rev = "16de68d283a366a03e359e66d37870980b3f7923";
    hash = "sha256-iw1tMGS0yCDf7XSVVIuC0n3lH/+t7A/mTPd9nIxr5Is=";
  };

  nativeBuildInputs = [ cmake ninja pkg-config ];
  buildInputs = [ openssl ] ++ lib.optionals rocmSupport (with rocmPackages; [
    clr hipblas rocblas
  ]);

  cmakeFlags = [
    (lib.cmakeBool "GGML_NATIVE" false)
    (lib.cmakeBool "GGML_CPU_ALL_VARIANTS" true)
    (lib.cmakeBool "GGML_BACKEND_DL" true)
    (lib.cmakeBool "GGML_HIP" rocmSupport)
    (lib.cmakeBool "LLAMA_BUILD_TESTS" false)
    (lib.cmakeBool "LLAMA_BUILD_EXAMPLES" false)
    (lib.cmakeBool "LLAMA_BUILD_SERVER" true)
    # Avoid fetching the web UI during the sandboxed build.
    (lib.cmakeBool "LLAMA_BUILD_UI" false)
    (lib.cmakeBool "LLAMA_USE_PREBUILT_UI" false)
    (lib.cmakeBool "LLAMA_BUILD_IS_DEV" false)
    (lib.cmakeFeature "LLAMA_BUILD_NUMBER" "1")
    (lib.cmakeFeature "LLAMA_BUILD_COMMIT" "16de68d")
  ] ++ lib.optionals rocmSupport [
    (lib.cmakeFeature "CMAKE_HIP_COMPILER" "${rocmPackages.clr.hipClangPath}/clang++")
    (lib.cmakeFeature "CMAKE_HIP_ARCHITECTURES" (lib.concatStringsSep ";" rocmGpuTargets))
  ];

  postInstall = ''
    for executable in "$out"/bin/llama*; do
      [ -f "$executable" ] || continue
      name="$(basename "$executable")"
      mv "$executable" "$out/bin/avalon''${name#llama}"
    done
  '';

  meta = {
    description = "Llama.cpp fork optimized for Qwen MoE inference";
    homepage = "https://github.com/freriadoc/Avalon";
    license = lib.licenses.mit;
    platforms = lib.platforms.linux;
    mainProgram = "avalon-server";
  };
}
