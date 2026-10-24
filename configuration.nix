# Edit this configuration file to define what should be installed on
# your system. Help is available in the configuration.nix(5) man page, on
# https://search.nixos.org/options and in the NixOS manual (`nixos-help`).

{ config, inputs, lib, pkgs, ... }:

let
  unstablePkgs = inputs.nixpkgs-unstable.legacyPackages.${pkgs.stdenv.hostPlatform.system};
  npu = import ./npu { inherit pkgs; };

  nixos-update = pkgs.writeShellApplication {
    name = "nixos-update";
    runtimeInputs = [ pkgs.git pkgs.nixos-rebuild ];
    text = ''
      if (( EUID != 0 )); then
        echo "nixos-update must be run as root (use: sudo nixos-update)" >&2
        exit 1
      fi

      config_dir=/etc/nixos

      if [[ ! -d "$config_dir/.git" ]]; then
        echo "$config_dir is not a Git checkout" >&2
        exit 1
      fi

      if [[ -n "$(git -C "$config_dir" status --porcelain)" ]]; then
        echo "$config_dir has uncommitted changes; refusing to update" >&2
        exit 1
      fi

      git -C "$config_dir" pull --ff-only
      exec nixos-rebuild switch --flake "$config_dir#nixos"
    '';
  };

  overrideVersionWhenLower = drv: version: fn:
    if builtins.compareVersions version drv.version == 1 then
      fn drv
    else
      drv;

  chatgpt = (overrideVersionWhenLower
    (pkgs.callPackage "${inputs.chatgpt-pr}/pkgs/by-name/ch/chatgpt/package.nix" { })
    "26.1007.71326"
    (d: d.overrideAttrs (_: {
      version = "26.1007.71326";
      src = pkgs.fetchurl {
        url = "https://persistent.oaistatic.com/codex-app-prod/linux/deb/pool/main/c/chatgpt/chatgpt_26.1007.71326_amd64.deb";
        hash = "sha256-DgBrSTFxL1qQ1M0ElyxuC1M87c39srUZRtW5NnRx9VY=";
      };
    }))).overrideAttrs (previous: {
      # New releases may omit the bundled LaTeX plugin.
      installPhase = lib.replaceStrings
        [ ''ln -sf ${lib.getExe pkgs.tectonic-unwrapped} "$out/lib/chatgpt/resources/plugins/openai-bundled/plugins/latex/bin/tectonic"'' ]
        [ ''
          if [ -d "$out/lib/chatgpt/resources/plugins/openai-bundled/plugins/latex/bin" ]; then
            ln -sf ${lib.getExe pkgs.tectonic-unwrapped} "$out/lib/chatgpt/resources/plugins/openai-bundled/plugins/latex/bin/tectonic"
          fi
        '' ]
        previous.installPhase;
      # Render natively on Wayland to avoid blurry fractional XWayland scaling.
      postFixup = (previous.postFixup or "") + ''
        wrapProgram "$out/bin/chatgpt" --add-flags "--ozone-platform=wayland"
      '';
    });

  llamaNpu = import ./npu/llama-prefill { inherit pkgs npu; };
  llamaBase = package: overrideVersionWhenLower package
    "0.7.0"
    (d: d.overrideAttrs (previous: {
      # Use the stable release until nixpkgs catches up.
      version = "0.7.0";
      src = pkgs.fetchFromGitHub {
        owner = "ggml-org";
        repo = "llama.cpp";
        tag = "v0.7.0";
        hash = "sha256-pEDYgI3AMTU+cekvRX7NNt5wddznydQSNuerswvEpsA=";
      };
      npmDepsHash = "sha256-hqYzdp1FwPhYkBP278fQXgY9NCKB6Q3P6Gko7BfP0m0=";
      preConfigure = lib.replaceStrings
        [ "LLAMA_BUILD_NUMBER=11146" ]
        [ "LLAMA_BUILD_NUMBER=1" ]
        previous.preConfigure;
      cmakeFlags = builtins.filter
        (flag: !(lib.hasPrefix "-DLLAMA_BUILD_NUMBER=" flag
          || lib.hasPrefix "-DLLAMA_BUILD_COMMIT=" flag
          || lib.hasPrefix "-DLLAMA_BUILD_IS_DEV=" flag))
        previous.cmakeFlags ++ [
          "-DLLAMA_BUILD_NUMBER=1"
          "-DLLAMA_BUILD_COMMIT=620d222"
          "-DLLAMA_BUILD_IS_DEV=OFF"
        ];
    }));
  llama-cpp-vulkan = (llamaNpu.patch (llamaBase unstablePkgs.llama-cpp-vulkan)).overrideAttrs (previous: {
    # Average Flash-Next hyper-connection streams for DFlash2/EAGLE3.
    # https://github.com/unslothai/llama.cpp/pull/307
    patches = (previous.patches or []) ++ [ ./patches/llama-qwen4exp-dflash-layer-input.patch ];
  });
  # Keep HIP separate so backend comparisons use explicit command names.
  llama-cpp-hip = (llamaBase (unstablePkgs.llama-cpp.override {
    rocmSupport = true;
    vulkanSupport = false;
    # ROCm 7.2 ships RDNA 3.5 BLAS kernels for gfx1150, but not gfx1152.
    rocmGpuTargets = [ "gfx1150" ];
  })).overrideAttrs (previous: {
    patches = (previous.patches or []) ++ [
      ./patches/llama-qwen4exp-dflash-layer-input.patch
      ./patches/llama-hip-hc-up.patch
      ./patches/llama-hip-dense-bf16.patch
      ./patches/llama-hip-q6-pair.patch
    ];
  });
  llama-hip = pkgs.runCommand "llama-hip" {
    nativeBuildInputs = [ pkgs.makeWrapper ];
    passthru.package = llama-cpp-hip;
  } ''
    mkdir -p $out/bin
    for command in llama llama-cli llama-server llama-bench; do
      makeWrapper ${llama-cpp-hip}/bin/$command $out/bin/$command-hip \
        --set-default HSA_OVERRIDE_GFX_VERSION 11.5.0
    done
  '';
  llama-npu = llamaNpu.launcher llama-cpp-vulkan;
in

{
  imports =
    [ # Include the results of the hardware scan.
      ./hardware-configuration.nix
    ];

  security.pam.loginLimits = [
    {
      domain = "fabian";   # your username
      type = "soft";
      item = "memlock";
      value = "unlimited";
    }
    {
      domain = "fabian";
      type = "hard";
      item = "memlock";
      value = "unlimited";
    }
  ];

  # Use the systemd-boot EFI boot loader.
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # Use latest kernel.
  boot.kernelPackages = pkgs.linuxPackages_latest;

  boot.kernelParams = [
    # Keep IOMMU enabled: the AMD XDNA NPU driver requires it.
    "amdgpu.gttsize=91136"
  ];

  # Alternate boot entry for comparing GPU performance without IOMMU.
  # The AMD XDNA NPU is unavailable in this specialisation.
  specialisation.iommu-off.configuration = {
    boot.kernelParams = [ "amd_iommu=off" ];
  };

  boot.extraModprobeConfig = ''
    options ttm pages_limit=32505856
    options ttm page_pool_size=32505856
  '';


  #boot.extraModprobeConfig = ''
  #  options ttm page_pool_size=17825792
  #  options ttm pages_limit=17825792
  #'';

  # networking.hostName = "nixos"; # Define your hostname.
  # Pick only one of the below networking options.
  # networking.wireless.enable = true;  # Enables wireless support via wpa_supplicant.
  networking.networkmanager.enable = true;  # Easiest to use and most distros use this by default.

  # Set your time zone.
  # time.timeZone = "Europe/Amsterdam";

  # Configure network proxy if necessary
  # networking.proxy.default = "http://user:password@proxy:port/";
  # networking.proxy.noProxy = "127.0.0.1,localhost,internal.domain";

  # Select internationalisation properties.
  # i18n.defaultLocale = "en_US.UTF-8";
  # console = {
  #   font = "Lat2-Terminus16";
  #   keyMap = "us";
  #   useXkbConfig = true; # use xkb.options in tty.
  # };

  # Enable the X11 windowing system.
  # services.xserver.enable = true;
  services.desktopManager.plasma6.enable = true;
  services.displayManager.sddm.enable = true;
  services.displayManager.autoLogin = {
    user = "fabian";
    enable = true;
  };

  services.tailscale.enable = true;


  # Configure keymap in X11
  # services.xserver.xkb.layout = "us";
  # services.xserver.xkb.options = "eurosign:e,caps:escape";

  # Enable CUPS to print documents.
  # services.printing.enable = true;

  # Enable sound.
  # services.pulseaudio.enable = true;
  # OR
  # services.pipewire = {
  #   enable = true;
  #   pulse.enable = true;
  # };

  # Enable touchpad support (enabled default in most desktopManager).
  # services.libinput.enable = true;

  # Define a user account. Don't forget to set a password with ‘passwd’.
  users.users.fabian = {
    isNormalUser = true;
    extraGroups = [ "wheel" ]; # Enable ‘sudo’ for the user.
    packages = with pkgs; [
      tree
    ];
  };

  virtualisation.waydroid.enable = true;
  virtualisation.waydroid.package = pkgs.waydroid.overrideAttrs {
    postInstall = ''
      ${pkgs.gnused}/bin/sed -i -e 's/LXC_USE_NFT="false"/LXC_USE_NFT="true"/' -e 's/-legacy//' $out/lib/waydroid/data/scripts/waydroid-net.sh
    '';
  };

  virtualisation.podman.enable = true;

  nixpkgs.config = {
    allowUnfreePredicate = pkg: builtins.elem (lib.getName pkg) [
      "android-studio" "vscode" "discord" "clion" "spotify" "claude-code" "cursor" "amp-cli" "idea" "chatgpt" "google-chrome" "zoom"
    ];

    permittedInsecurePackages = [
      "electron-39.8.10"
    ];

  };

  environment.etc."systemd/system-sleep/fingerprint-reset" = {
    mode = "0550";
    text = ''
      #!${pkgs.stdenv.shell}
      PATH=${pkgs.coreutils}/bin
      PCI_FUNC="0000:c1:00.4"
      GOODIX_ID="27c6:609c"
      DRIVER_PATH="/sys/bus/pci/drivers/xhci_hcd"

      case $1 in
        post)
        sleep 2

        if ! ${pkgs.usbutils}/bin/lsusb -d "$GOODIX_ID" >/dev/null 2>&1; then
          echo "fingerprint sensor gone, rebinding"
          echo "$PCI_FUNC" > "$DRIVER_PATH/unbind"
          sleep 1
          echo "$PCI_FUNC" > "$DRIVER_PATH/bind"
        else
          echo "fingerprint still active after $2"
        fi
        ;;
      esac
    '';
  };

  # programs.firefox.enable = true;


  #services.power-profiles-daemon.enable = false;

  # List packages installed in system profile.
  # You can use https://search.nixos.org/ to find more packages (and options).
  environment.systemPackages = with pkgs; [
    npu.runtime
    npu.tools
    amp-cli
    vim
    # jetbrains.clion
    gparted
    wget
    kdePackages.kdevelop
    vscode
    android-studio
    firefox
    google-chrome
    htop
    perf
    hotspot
    python3
    (gnucash.overrideAttrs (prev: {
      preFixup = prev.preFixup + ''gappsWrapperArgs+=(--set LANGUAGE de_DE.UTF-8)'';
    }))
    telegram-desktop
    zoom-us
    ncdu
    code-cursor
    glab
    signal-desktop
    logseq
    # (import ./audacity.nix { inherit pkgs; })
    git
    git-lfs
    vlc
    (keepassxc.overrideAttrs (finalAttrs: prevAttrs: {
      version = "331a2de136398f733136c51f1badae5d154878bc";
      src = prevAttrs.src.override {
        hash = "sha256-deIg59jSCW5e0WZ5nCTkV46ZK6cwjneA1+3nLlzekE4=";
        rev = finalAttrs.version;
        tag = null;
      };

      patches = prevAttrs.patches ++ [
        (pkgs.writeText "botan.patch" ''
          diff --git a/src/sshagent/OpenSSHKeyGen.cpp b/src/sshagent/OpenSSHKeyGen.cpp
          index a3d88807fb..6212cd0e9d 100644
          --- a/src/sshagent/OpenSSHKeyGen.cpp
          +++ b/src/sshagent/OpenSSHKeyGen.cpp
          @@ -24,6 +24,10 @@
           #include <botan/ecdsa.h>
           #include <botan/ed25519.h>
           #include <botan/rsa.h>
          +#include <botan/version.h>
          +#if BOTAN_VERSION_CODE >= BOTAN_VERSION_CODE_FOR(3,11,0)
          +  #include <botan/ec_group.h>
          +#endif

           namespace OpenSSHKeyGen
           {
        '')
      ];

      buildInputs = prevAttrs.buildInputs ++ [keyutils];
    }))
    calibre
    #rustc
    rustup
    cargo
    zig
    nix-index
    nextcloud-client
    binutils
    gdb
    keepass-diff
    kdePackages.partitionmanager
    gnumake
    calibre
    unstablePkgs.codex
    unstablePkgs.t3code
    chatgpt
    (pkgs.callPackage ./pi-gui.nix { })
    gh
    nixd
    nixos-update
    libreoffice
    scribus
    lilypond
    cmake
    ghostty
    zed-editor
    texlab
    lldb
    jetbrains.idea
    zls
    thunderbird
    wireshark
    kdePackages.pim-sieve-editor
    kdePackages.kdenlive
    element-desktop
    discord
    llama-cpp-vulkan
    llama-hip
    llama-npu
    ntfs3g
    spotify
    claude-code
    inputs.hermes-agent.packages.${pkgs.stdenv.hostPlatform.system}.desktop
  ];

   services.pipewire.wireplumber.extraConfig.no-ucm = {
     "monitor.alsa.properties" = {
       "alsa.use-ucm" = false;
     };
   };

  hardware.bluetooth.enable = true;

  services.fprintd.enable = true;
  services.ollama.enable = true;

  programs.java.enable = true;
  programs.kdeconnect.enable = true;

  nix.settings = {
   experimental-features = ["nix-command" "flakes"];
  };

  i18n.extraLocales = [ "de_DE.UTF-8/UTF-8" ];

  # Some programs need SUID wrappers, can be configured further or are
  # started in user sessions.
  # programs.mtr.enable = true;
  # programs.gnupg.agent = {
  #   enable = true;
  #   enableSSHSupport = true;
  # };

  # List services that you want to enable:

  # Enable the OpenSSH daemon.
  # services.openssh.enable = true;

  # Open ports in the firewall.
  networking.firewall.allowedTCPPorts = [ 8080 ];
  # networking.firewall.allowedUDPPorts = [ ... ];
  # Or disable the firewall altogether.
  # networking.firewall.enable = false;

  # Copy the NixOS configuration file and link it from the resulting system
  # (/run/current-system/configuration.nix). This is useful in case you
  # accidentally delete configuration.nix.
  # system.copySystemConfiguration = true;

  # This option defines the first version of NixOS you have installed on this particular machine,
  # and is used to maintain compatibility with application data (e.g. databases) created on older NixOS versions.
  #
  # Most users should NEVER change this value after the initial install, for any reason,
  # even if you've upgraded your system to a new NixOS release.
  #
  # This value does NOT affect the Nixpkgs version your packages and OS are pulled from,
  # so changing it will NOT upgrade your system - see https://nixos.org/manual/nixos/stable/#sec-upgrading for how
  # to actually do that.
  #
  # This value being lower than the current NixOS release does NOT mean your system is
  # out of date, out of support, or vulnerable.
  #
  # Do NOT change this value unless you have manually inspected all the changes it would make to your configuration,
  # and migrated your data accordingly.
  #
  # For more information, see `man configuration.nix` or https://nixos.org/manual/nixos/stable/options#opt-system.stateVersion .
  system.stateVersion = "25.05"; # Did you read the comment?

}
