{ pkgs }:

let
  ovPlugins = pkgs.fetchFromGitHub {
    owner = "intel";
    repo = "openvino-plugins-ai-audacity";
    rev = "v3.7.1-R4.2";
    hash = "sha256-nIW55AVMwttUdAK95GpYMrK3nQRK2yiDZm6ePiCLXI0=";
  };

  whisperCppOV = pkgs.whisper-cpp.overrideAttrs (old: {
    cmakeFlags = (old.cmakeFlags or []) ++ [
      "-DWHISPER_OPENVINO=ON"
    ];

    buildInputs = (old.buildInputs or []) ++ [
      pkgs.openvino
    ];
  });
in

pkgs.audacity.overrideAttrs (old: {
  pname = "audacity-openvino";

  nativeBuildInputs =
    (old.nativeBuildInputs or [])
    ++ [ pkgs.cmake pkgs.pkg-config ];

  buildInputs =
    (old.buildInputs or [])
    ++ [
      pkgs.onnxruntime
      pkgs.libtorch-bin
      pkgs.openvino
      whisperCppOV
    ];

  postPatch = (old.postPatch or "") + ''
    cp -r ${ovPlugins}/mod-openvino modules/

    echo 'add_subdirectory(mod-openvino)' \
      >> modules/CMakeLists.txt
  '';

  cmakeFlags = (old.cmakeFlags or []) ++ [
    "-DOpenVINO_DIR=${pkgs.openvino}/runtime/cmake"
  ];
})
