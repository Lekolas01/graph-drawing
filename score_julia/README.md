# Setup 
Install julia by running the following command [https://julialang.org/downloads/](https://julialang.org/downloads/)
```bash
curl -fsSL https://install.julialang.org | sh
```

# Execution
Inside this directory, run 
```bash
julia --project=. score.jl
```

`--project=.` points the julia executable to the directory containing the `Manifest.toml` and `Project.toml` files.
These two contain the dependencies and metadata about the project. 