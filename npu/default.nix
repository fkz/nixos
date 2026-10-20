# Packaging adapted from noamsto/nix-amd-ai at
# 8e1fd17dcc61bd179a90433109d5c5fa45631ff1 (MIT; see LICENSE.packaging).
{ pkgs }:
let
  xrt = pkgs.callPackage ./xrt.nix { };
  plugin = pkgs.callPackage ./xdna-plugin.nix { inherit xrt; };
  peano = pkgs.callPackage ./llvm-aie.nix { };
  iron = pkgs.callPackage ./mlir-aie.nix { llvm-aie = peano; };
  runtime = pkgs.symlinkJoin {
    name = "amd-npu-runtime";
    paths = [ xrt plugin ];
    nativeBuildInputs = [ pkgs.makeWrapper ];
    postBuild = ''
      # XRT locates its driver plugins relative to XILINX_XRT.
      wrapProgram $out/bin/xrt-smi \
        --set XILINX_XRT $out/opt/xilinx/xrt \
        --prefix LD_LIBRARY_PATH : $out/opt/xilinx/xrt/lib
    '';
  };
  tools = pkgs.writeShellApplication {
    name = "npu-python";
    runtimeInputs = [ runtime iron peano pkgs.clang pkgs.lld ];
    text = ''
      export XILINX_XRT=${runtime}/opt/xilinx/xrt
      export LD_LIBRARY_PATH=${runtime}/opt/xilinx/xrt/lib:''${LD_LIBRARY_PATH:-}
      export PYTHONPATH=${iron.pythonPath}:${runtime}/opt/xilinx/xrt/python:''${PYTHONPATH:-}
      export PEANO_INSTALL_DIR=${peano}
      exec ${iron.python}/bin/python3.12 "$@"
    '';
  };
in {
  inherit xrt plugin peano iron runtime tools;
}
