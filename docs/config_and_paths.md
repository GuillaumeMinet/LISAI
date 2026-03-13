### Configuration & Path Architecture

- **Settings** loads and validates YAML configuration and builds the template context.
- **Paths** resolves all filesystem locations from configuration templates.
- **Runtime specs** (e.g. RunSpec/ModelSpec) provide typed, run-specific views used by training/inference.
- The goal is a **single source of truth**, zero hardcoded paths, and clear separation between config, infra, and runtime.
