# Avalon

The source is pinned in `avalon.nix`. Build the CPU package with
`nix build .#avalon`, or the HIP package for this machine with
`nix build .#avalon-hip`.

Run the HIP server with:

```sh
nix run .#avalon-hip -- -m /path/to/model.gguf --host 127.0.0.1 --port 8080
```

Installed executables use the `avalon` prefix (`avalon-server`, `avalon-cli`,
`avalon-bench`, etc.) to coexist with llama.cpp. The embedded web UI is disabled
to keep the build independent of downloaded UI assets; the server API remains
available. Models are supplied separately.

The GPU reports gfx1152. Like the existing llama HIP package, `avalon-hip`
compiles for gfx1150 and sets `HSA_OVERRIDE_GFX_VERSION=11.5.0` by default,
matching the available ROCm BLAS kernels. This can be overridden through the
environment. `avalon-hip-unwrapped` exposes the build without that launcher.
Other GPUs can use `avalon.override { rocmSupport = true;
rocmGpuTargets = [ "gfx1100" ]; }` with the appropriate target.

The HIP package is included in the system package list, making `avalon-server`,
`avalon-cli`, and the other prefixed commands available after activation.
