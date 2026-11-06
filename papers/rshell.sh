#!/usr/bin/env bash
# Run a command with R + the packages make_figures.R needs (fetched by nix, nothing installed globally).
#   bash papers/rshell.sh Rscript papers/make_figures.R
exec nix shell --impure --expr 'let p = (builtins.getFlake "nixpkgs").legacyPackages.${builtins.currentSystem}; in p.rWrapper.override { packages = with p.rPackages; [ ggplot2 dplyr readr tidyr scales patchwork ]; }' -c "$@"
