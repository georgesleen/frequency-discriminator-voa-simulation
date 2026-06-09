{
  description = "Photonics Lab simulation port - DEVSIM + femwell devshell (FHS)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      # DEVSIM's docs explicitly recommend MKL; its bundled UMFPACK was
      # written against MKL's BLAS conventions and miscalls OpenBLAS's DGER
      # (parameter 9 / LDA bug). MKL is unfree (Intel SSL) - narrow the
      # exception to just that package rather than blanket-allowing unfree.
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfreePredicate = pkg:
          builtins.elem (nixpkgs.lib.getName pkg) [ "mkl" ];
      };

      # A real FHS-compliant chroot: /usr/lib, /lib64 etc. exist with normal
      # libraries. That's what pip wheels built for manylinux assume, and
      # it's how DEVSIM's runtime dlopen("libopenblas.so") resolves cleanly -
      # no LD_LIBRARY_PATH gymnastics, no BLAS-symbol-resolution surprises.
      # nix-ld can't help here because it only patches ELF interpreters at
      # exec time; once Python is running, dlopen() is back on its own.
      fhs = pkgs.buildFHSEnv {
        name = "devsim-shell";
        targetPkgs = pkgs: with pkgs; [
          uv
          python312
          mkl              # libmkl_rt.so - DEVSIM's expected BLAS/LAPACK
          gmsh

          # Libs that DEVSIM's compiled extensions and pip wheels routinely
          # dlopen by bare soname; in FHS they resolve via /usr/lib.
          stdenv.cc.cc.lib   # libstdc++.so.6
          zlib
          expat
          libGL
          libGLU
          libx11
          libxrender
          libxcursor
          libxinerama
          libxfixes
          libxext
          libxft
          fontconfig
          freetype
        ];
        profile = ''
          export DEVSIM_MATH_LIBS=libmkl_rt.so
          export UV_PYTHON=${pkgs.python312}/bin/python3.12
        '';
        runScript = "bash";
      };
      # Lightweight shell for direnv / IDE use: tools on PATH, env vars set,
      # but no bwrap namespace. DEVSIM still requires `nix run` (FHS env).
      devShell = pkgs.mkShell {
        packages = with pkgs; [ uv python312 gmsh ];
        shellHook = ''
          export DEVSIM_MATH_LIBS=libmkl_rt.so
          export UV_PYTHON=${pkgs.python312}/bin/python3.12
        '';
      };
    in {
      # Default devShell is bwrap-free so direnv enter/exit works without
      # namespace nesting errors. Use `nix develop .#fhs` for a full FHS shell.
      devShells.${system} = {
        default = devShell;
        fhs = fhs.env;
      };

      # Fast unit tests as a flake check: the pure physics formulas in
      # src/physics.py need only numpy, so a plain nixpkgs Python runs them in
      # the build sandbox - no uv, no MKL, no network. The DEVSIM integration
      # tests can't run here (bwrap-FHS + MKL + uv want namespaces/network the
      # sandbox forbids); CI runs those via `nix run` instead.
      checks.${system}.unit = pkgs.runCommand "unit-tests" {
        nativeBuildInputs = [
          (pkgs.python312.withPackages (ps: with ps; [ numpy pytest ]))
        ];
      } ''
        export HOME=$TMPDIR
        export PYTHONDONTWRITEBYTECODE=1
        cd ${self}
        python -m pytest tests/test_physics.py -q -p no:cacheprovider
        touch $out
      '';

      # `nix run . -- -c "command"` -> scripted invocation inside FHS.
      apps.${system}.default = {
        type = "app";
        program = "${fhs}/bin/devsim-shell";
      };

      packages.${system}.default = fhs;
    };
}
