# Build the pi-gui desktop application from the pinned thinking/timing fork.
# Instantiate with:
#   nix-build --expr 'let pkgs = import <nixpkgs> {}; in pkgs.callPackage ./pi-gui.nix {}'
{
  lib,
  stdenv,
  fetchFromGitHub,
  fetchPnpmDeps,
  nodejs_22,
  pnpm_10,
  pnpmConfigHook,
  # Upstream pins Electron 37.10.3. Supply a compatible Electron 37 build here.
  electron,
  makeWrapper,
  makeDesktopItem,
  git,
  openssh,
}:

let
  pname = "pi-gui";
  version = "1.0.1";
  pnpm = pnpm_10;

  src = fetchFromGitHub {
    owner = "fkz";
    repo = "pi-gui";
    rev = "ba95d47ee6fb03dcbb7c68c0ff4c1218db2a114f";
    hash = "sha256-i+lqjSPYy9jTF41AX68XQdSx3LeY4zOAkpm//iv8jTA=";
  };

  desktopItem = makeDesktopItem {
    name = pname;
    desktopName = "pi-gui";
    comment = "Desktop app for the pi coding agent";
    exec = pname;
    categories = [ "Development" ];
    terminal = false;
  };
in
stdenv.mkDerivation (finalAttrs: {
  inherit pname version src;

  pnpmDeps = fetchPnpmDeps {
    inherit (finalAttrs) pname version src;
    inherit pnpm;
    fetcherVersion = 4;
    pnpmWorkspaces = [
      "@pi-gui/desktop"
      "@pi-gui/session-driver"
      "@pi-gui/pi-sdk-driver"
      "@pi-gui/extension-ui"
      "@pi-gui/catalogs"
    ];
    hash = "sha256-dq44ObnohNkir52lnjyWjKBSYLz/WGWDoL3c9prkqtU=";
  };

  nativeBuildInputs = [
    nodejs_22
    pnpm
    pnpmConfigHook
    makeWrapper
  ];

  buildPhase = ''
    runHook preBuild
    pnpm --filter @pi-gui/desktop run build
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/lib/${pname}" "$out/bin" "$out/share/applications"
    cp -a . "$out/lib/${pname}/"

    makeWrapper ${lib.getExe electron} "$out/bin/${pname}" \
      --add-flags "$out/lib/${pname}/apps/desktop" \
      --prefix PATH : ${lib.makeBinPath [ git openssh ]}

    install -Dm444 ${desktopItem}/share/applications/${pname}.desktop \
      "$out/share/applications/${pname}.desktop"

    runHook postInstall
  '';

  meta = {
    description = "Desktop app for the pi coding agent, built from source";
    homepage = "https://github.com/fkz/pi-gui";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = pname;
  };
})
