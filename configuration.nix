# Edit this configuration file to define what should be installed on
# your system. Help is available in the configuration.nix(5) man page, on
# https://search.nixos.org/options and in the NixOS manual (`nixos-help`).

{ config, lib, pkgs, ... }:

{
  imports =
    [ # Include the results of the hardware scan.
      ./hardware-configuration.nix
    ];

  # Use the systemd-boot EFI boot loader.
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # Use latest kernel.
  boot.kernelPackages = pkgs.linuxPackages_latest;

  boot.extraModprobeConfig = ''
    options ttm page_pool_size=17825792
    options ttm pages_limit=17825792
  '';

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

  nixpkgs.config = {
    allowUnfreePredicate = pkg: builtins.elem (lib.getName pkg) [
      "android-studio" "discord" "clion" "spotify" "claude-code" "cursor"
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
    vim
    jetbrains.clion
    gparted
    wget
    kdePackages.kdevelop
    #vscode
    android-studio
    firefox
    htop
    (gnucash.overrideAttrs (prev: {
      preFixup = prev.preFixup + ''gappsWrapperArgs+=(--set LANGUAGE de_DE.UTF-8)'';
    }))
    ncdu
    code-cursor
    glab
    signal-desktop
    git
    git-lfs
    (keepassxc.overrideAttrs (finalAttrs: prevAttrs: {
      version = "331a2de136398f733136c51f1badae5d154878bc";
      src = prevAttrs.src.override {
        hash = "sha256-deIg59jSCW5e0WZ5nCTkV46ZK6cwjneA1+3nLlzekE4=";
        rev = finalAttrs.version;
        tag = null;
      };

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
    kdePackages.partitionmanager
    gnumake
    calibre
    libreoffice
    cmake
    zed-editor
    lldb
    jetbrains.idea-oss
    zls
    thunderbird
    kdePackages.pim-sieve-editor
    kdePackages.kdenlive
    element-desktop
    discord
    llama-cpp-vulkan
    ntfs3g
    spotify
    claude-code
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
  # networking.firewall.allowedTCPPorts = [ ... ];
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

