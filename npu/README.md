# AMD NPU development tools

The mainline `amdxdna` driver and firmware are supplied by the existing NixOS
kernel and firmware packages. IOMMU must remain enabled to use the NPU.

This directory pins XRT, the XDNA userspace plugin, MLIR-AIE/IRON 1.4.2,
and its matching Peano compiler. Packaging is adapted from
https://github.com/noamsto/nix-amd-ai/tree/8e1fd17dcc61bd179a90433109d5c5fa45631ff1
under the accompanying MIT license.

Build and use without activating the system:

```sh
nix build /home/fabian/nixos#npu-runtime -o /tmp/npu-runtime
/tmp/npu-runtime/bin/xrt-smi examine
nix build /home/fabian/nixos#npu-tools -o /tmp/npu-tools
/tmp/npu-tools/bin/npu-python your_kernel.py
```

`npu-python` supplies the matching Python packages, compiler, runtime libraries,
and XRT driver plugin. It does not replace the kernel driver or firmware.
The configuration also installs these launchers system-wide after activation
with `sudo nixos-update`.
