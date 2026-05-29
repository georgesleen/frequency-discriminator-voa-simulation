{
  description = "Photonics Lab simulation port — DEVSIM + femwell devshell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # pip wheels are pre-compiled for FHS Linux, so their C extensions
      # dlopen() libraries at paths NixOS doesn't provide by default.
      # nix-ld only patches ELF interpreters (exec'd binaries); it doesn't
      # intercept dlopen() calls made inside an already-running nix Python.
      # Gathered with:  ldd .venv/**/*.so 2>/dev/null | grep "not found"
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib   # libstdc++.so.6     — devsim, scipy
        openblas            # libopenblas.so     — DEVSIM BLAS/LAPACK
        zlib                # libz.so.1          — numpy, scipy, PIL
        expat               # libexpat.so.1      — gmsh, several wheels
        libGL               # libGL.so.1         — vtk
        libGLU              # libGLU.so.1        — gmsh Python bindings
        libx11              # libX11.so.6        — vtk, matplotlib X
        libxrender          # libXrender.so.1    — vtk
        libxcursor          # libXcursor.so.1    — gmsh/fltk (DT_RUNPATH in
        libxinerama         # libXinerama.so.1   #  fltk not honoured when
        libxfixes           # libXfixes.so.3     #  LD_LIBRARY_PATH is set;
        libxext             # libXext.so.6       #  list them explicitly)
        libxft              # libXft.so.2        — fltk text rendering
        fontconfig          # libfontconfig.so.1 — fltk font lookup
        freetype            # libfreetype.so.6   — fltk/fontconfig
        # gmsh: pip's find_library("gmsh") searches LD_LIBRARY_PATH
        gmsh
      ];
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          uv
          python312
          openblas
          gmsh
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH
          export DEVSIM_MATH_LIBS=libopenblas.so
          export UV_PYTHON=${pkgs.python312}/bin/python3.12
        '';
      };
    };
}
