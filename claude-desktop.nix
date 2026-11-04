{ lib, stdenvNoCC, fetchurl, dpkg, buildFHSEnv, makeWrapper, runCommand }:

let
  version = "4.16336.0";
  desktop = stdenvNoCC.mkDerivation {
    pname = "claude-desktop-unwrapped";
    inherit version;
    src = fetchurl {
      url = "https://downloads.claude.ai/claude-desktop/apt/stable/pool/main/c/claude-desktop/claude-desktop_${version}_amd64.deb";
      hash = "sha256-OBBzPpN5L3Dq6SEXXRPlPRzGvc6dwjEX8FiAh1IaZLw=";
    };
    nativeBuildInputs = [ dpkg ];
    unpackPhase = "dpkg-deb --fsys-tarfile $src | tar -x";
    installPhase = ''
      mkdir -p $out
      cp -r usr/lib usr/share $out/
      rm -rf $out/share/{doc,lintian}
    '';
    dontFixup = true;
  };

  # The app downloads native Claude Code tools at runtime. Give those tools,
  # as well as the bundled Electron runtime, the standard Linux loader paths.
  runtime = buildFHSEnv {
    name = "claude-desktop";
    targetPkgs = p: with p; [
      alsa-lib at-spi2-core cairo cups dbus expat glib gtk3
      libdrm libgbm libGL libnotify libsecret libuuid libxcb libxkbcommon
      libx11 libxcomposite libxdamage libxext libxfixes libxrandr libxtst
      nspr nss pango pipewire systemd libpulseaudio libappindicator-gtk3
      stdenv.cc.cc zlib
      bash coreutils findutils gnugrep gnused git openssh bubblewrap socat
      xdg-utils cacert
    ];
    runScript = "${desktop}/lib/claude-desktop/claude-desktop";
  };
in
runCommand "claude-desktop-${version}" {
  nativeBuildInputs = [ makeWrapper ];
  meta = {
    description = "Official Claude Desktop with Claude Code (Linux beta)";
    homepage = "https://claude.com/download";
    license = lib.licenses.unfree;
    platforms = [ "x86_64-linux" ];
    mainProgram = "claude-desktop";
  };
} ''
  mkdir -p $out/bin
  makeWrapper ${runtime}/bin/claude-desktop $out/bin/claude-desktop \
    --add-flags "--ozone-platform-hint=auto"
  ln -s ${desktop}/share $out/share
''
