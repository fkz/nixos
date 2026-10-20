{ pkgs, npu }:
let
  kernels = pkgs.stdenvNoCC.mkDerivation {
    pname = "llama-npu-prefill-kernels";
    version = "1";
    src = pkgs.lib.cleanSourceWith {
      src = ./.;
      name = "llama-npu-kernel-source";
      filter = path: type: type == "directory" || builtins.elem (baseNameOf path) [ "design.py" "iq3_xxs.cc" "iq3_xxs_lut.h" "kernel.cc" "compile.py" ];
    };
    nativeBuildInputs = [ npu.tools ];
    dontConfigure = true;
    buildPhase = ''
      export NPU_CACHE_HOME=$TMPDIR/npu-cache
      npu-python compile.py
    '';
    installPhase = ''mkdir -p $out; cp artifacts/*.bin artifacts/*.xclbin $out/'';
  };
in {
  inherit kernels;
  patch = package: package.overrideAttrs (old: {
    patches = (old.patches or []) ++ [ ./llama-vulkan.patch ];
    postPatch = (old.postPatch or "") + ''
      cp ${./ggml-npu-prefill.inc} ggml/src/ggml-vulkan/ggml-npu-prefill.inc
    '';
    buildInputs = (old.buildInputs or []) ++ [ npu.xrt pkgs.libuuid ];
    nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ pkgs.makeWrapper ];
    cmakeFlags = (old.cmakeFlags or []) ++ [
      "-DGGML_VULKAN_NPU_PREFILL=ON"
      "-DGGML_NPU_XRT_INCLUDE=${npu.xrt}/opt/xilinx/xrt/include"
      "-DGGML_NPU_XRT_LIBRARY=${npu.xrt}/opt/xilinx/xrt/lib/libxrt_coreutil.so"
    ];
    postFixup = (old.postFixup or "") + ''
      for command in llama llama-server llama-bench; do
        if [ -x "$out/bin/$command" ]; then
          wrapProgram "$out/bin/$command" \
            --set-default XILINX_XRT ${npu.runtime}/opt/xilinx/xrt \
            --prefix LD_LIBRARY_PATH : ${npu.runtime}/opt/xilinx/xrt/lib \
            --set-default GGML_VK_NPU_KERNEL_DIR ${kernels}
        fi
      done
    '';
  });
  launcher = llama: pkgs.writeShellApplication {
    name = "llama-npu";
    runtimeInputs = [ llama ];
    text = ''
      export GGML_VK_NPU_PREFILL=1
      exec llama "$@"
    '';
  };
}
