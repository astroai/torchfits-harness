# Harness playbook

- id: subject
  desc: Install a torchfits wheel before benches or parity tests. pixi run dev installs this package with --no-deps. This repo does not compile torchfits or vendor CFITSIO.

- id: local-bench-env
  desc: Local bench tasks run in the default Pixi env. Comparators are already dependencies. Do not recreate a torchfits bench-all env or bench-gpu-install here.

- id: source-probes
  desc: Tests that read torchfits cpp_src, extern/cfitsio, or tests/cpp stay in the torchfits repo.
