{
  description = "Photonics Lab simulation port — DEVSIM + femwell devshell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          uv
          python312
          openblas
          gmsh
        ];

        # DEVSIM dlopens BLAS/LAPACK at runtime — point it at the Nix-store
        # openblas (which provides both BLAS and LAPACK symbols).
        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.openblas}/lib:$LD_LIBRARY_PATH
          export DEVSIM_MATH_LIBS=libopenblas.so
          export UV_PYTHON=${pkgs.python312}/bin/python3.12
        '';
      };
    };
}
