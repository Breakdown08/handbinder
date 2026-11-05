{
  description = "handbinder development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      # opencv-python from PyPI links against system libraries NixOS does not expose globally.
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.python314
          pkgs.uv
        ];
        # The interpreter comes from the same nixpkgs as the libraries below, so their glibc versions agree.
        UV_PYTHON = "${pkgs.python314}/bin/python3";
        UV_PYTHON_PREFERENCE = "only-system";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
          with pkgs;
          [
            dbus
            fontconfig
            freetype
            glib
            libGL
            libice
            libsm
            libx11
            libxcb
            libxext
            libxkbcommon
            stdenv.cc.cc.lib
            zlib
          ]
        );
        QT_QPA_PLATFORM = "xcb"; # the Qt bundled with opencv-python ships only this plugin
      };
    };
}
