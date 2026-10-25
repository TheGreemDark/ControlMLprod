"""Generate a hyperparameter grid and run each experiment in its own container."""

import argparse
import itertools
import re
import subprocess
from pathlib import Path
from typing import cast

import yaml

from controlmlprod.data.config import DATA_DIR, PROJECT_ROOT


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a YAML mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} must contain string keys")
    return cast(dict[str, object], value)


def _set_nested(config: dict[str, object], path: str, value: object) -> None:
    """Set a nested configuration value using a dotted path."""
    parts = path.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Invalid grid parameter path: {path!r}")

    current = config
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise TypeError(f"Cannot resolve grid parameter path: {path}")
        current = cast(dict[str, object], child)

    current[parts[-1]] = value


def _slug(value: object) -> str:
    """Convert a grid value into a path-safe run-name component."""
    if isinstance(value, float):
        raw = format(value, "g")
    elif isinstance(value, list):
        raw = "_".join(_slug(item) for item in value)
    else:
        raw = str(value)

    result = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    if not result:
        raise ValueError(f"Cannot create a run-name component from {value!r}")
    return result


def _resolve_base_config(grid_path: Path, base_name: str) -> Path:
    base_path = Path(base_name)
    if base_path.is_absolute():
        return base_path

    relative_to_grid = grid_path.parent / base_path
    if relative_to_grid.is_file():
        return relative_to_grid

    return PROJECT_ROOT / base_path


def build_grid_configs(grid_path: Path) -> list[dict[str, object]]:
    """Build all individual configs from a grid YAML file."""
    if not grid_path.is_absolute():
        grid_path = PROJECT_ROOT / grid_path

    grid_doc = _mapping(
        yaml.safe_load(grid_path.read_text(encoding="utf-8")),
        "grid config",
    )

    base_name = grid_doc.get("base")
    experiment_name = grid_doc.get("experiment_name")
    if not isinstance(base_name, str) or not base_name.strip():
        raise ValueError("Grid config must define a non-empty base path")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("Grid config must define experiment_name")

    base_path = _resolve_base_config(grid_path, base_name)
    base_config = _mapping(
        yaml.safe_load(base_path.read_text(encoding="utf-8")),
        "base config",
    )

    grid = _mapping(grid_doc.get("grid"), "grid")
    if not grid:
        raise ValueError("Grid must contain at least one parameter")

    parameter_names = list(grid)
    parameter_values: list[list[object]] = []

    for name in parameter_names:
        values = grid[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid parameter {name!r} must have a non-empty list")
        parameter_values.append(cast(list[object], values))

    model = _mapping(base_config.get("model"), "base.model")
    _mapping(base_config.get("train"), "base.train")
    model_type = model.get("type")
    if not isinstance(model_type, str):
        raise TypeError("base.model.type must be a string")

    model_prefix = {
        "simple_cnn": "cnn",
        "linear": "linear",
    }.get(model_type, _slug(model_type))

    configs: list[dict[str, object]] = []

    for combination in itertools.product(*parameter_values):
        candidate = yaml.safe_load(yaml.safe_dump(base_config))
        candidate = _mapping(candidate, "generated config")

        for name, value in zip(parameter_names, combination, strict=True):
            _set_nested(candidate, name, value)

        candidate["experiment_name"] = experiment_name

        candidate_train = _mapping(candidate.get("train"), "generated.train")
        candidate_model = _mapping(candidate.get("model"), "generated.model")
        candidate_seed = candidate.get("seed")
        candidate_type = candidate_model.get("type")

        if not isinstance(candidate_seed, int) or isinstance(candidate_seed, bool):
            raise TypeError("Generated config seed must be an integer")
        if not isinstance(candidate_type, str):
            raise TypeError("Generated model type must be a string")

        prefix = {
            "simple_cnn": "cnn",
            "linear": "linear",
        }.get(candidate_type, _slug(candidate_type))

        parts = [prefix]
        labels = {
            "train.lr": "lr",
            "train.batch_size": "bs",
        }
        for name, value in zip(parameter_names, combination, strict=True):
            label = labels.get(name, _slug(name.replace(".", "-")))
            parts.append(f"{label}{_slug(value)}")

        parts.append(f"seed{candidate_seed}")
        run_name = "-".join(parts)

        # Check that generated names use the final model configuration.
        if candidate_model.get("type") != model_type:
            run_name = run_name.replace(model_prefix, prefix, 1)

        candidate["run_name"] = run_name

        # Ensure the base train mapping is valid before writing the config.
        _mapping(candidate_train, "generated.train")
        configs.append(candidate)

    run_names = [config["run_name"] for config in configs]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Grid generated duplicate run names")

    return configs


def run_grid(grid_path: Path) -> None:
    """Save each config and start it in its own Docker container."""
    configs = build_grid_configs(grid_path)
    config_dir = DATA_DIR / "experiment_configs"

    for index, config in enumerate(configs, start=1):
        run_name = str(config["run_name"])
        experiment_name = str(config["experiment_name"])
        output_dir = config_dir / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / f"{run_name}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        container_config_path = (
            Path("/app/data/experiment_configs") / experiment_name / f"{run_name}.yaml"
        )

        print(
            f"[{index}/{len(configs)}] Starting {run_name}",
            flush=True,
        )

        subprocess.run(
            [
                "docker",
                "compose",
                "run",
                "--rm",
                "--no-deps",
                "--entrypoint",
                "python",
                "train",
                "-m",
                "controlmlprod.experiments.train",
                "--config",
                str(container_config_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the hyperparameter grid YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated run configs without starting containers",
    )
    args = parser.parse_args()

    configs = build_grid_configs(args.config)

    if args.dry_run:
        for index, config in enumerate(configs, start=1):
            print(
                f"{index}. {config['run_name']}: {yaml.safe_dump(config, sort_keys=False).strip()}"
            )
        print(f"Total runs: {len(configs)}")
        return

    run_grid(args.config)


if __name__ == "__main__":
    main()
