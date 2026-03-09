{
  inputs = {
    # This is pointing to an unstable release.
    # If you prefer a stable release instead, you can this to the latest number shown here: https://nixos.org/download
    # i.e. nixos-24.11
    # Use `nix flake update` to update the flake to the latest revision of the chosen release channel.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixos-hardware.url = "github:fkz/nixos-hardware/master";
  };
  outputs = inputs@{ self, nixpkgs, nixos-hardware, ... }: {
    # NOTE: 'nixos' is the default hostname
    nixosConfigurations.nixos = nixpkgs.lib.nixosSystem {
      modules = [ ./configuration.nix ]; #nixos-hardware.nixosModules.framework-amd-ai-300-series ];
    };
  };
}

