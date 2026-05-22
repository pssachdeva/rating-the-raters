import argparse
from pathlib import Path
import subprocess

from mhs_llms.config import load_severity_decomposition_config
from mhs_llms.facets import process_severity_decomposition_run, run_severity_decomposition_facets


DEFAULT_CONFIG_PATH = Path("configs/full_set_all_models/facets_severity_decomposition.yaml")
DEFAULT_OUTPUT_PATH = Path("data/full_run_models_reference_set_severity_decomposition_bias_terms.csv")
DEFAULT_VM_NAME = "Windows 11 (1)"
DEFAULT_WINDOWS_PROJECT_PATH = r"\\Mac\Home\projects\measuring_hate_speech_llms"
DEFAULT_FACETS_EXE = r"C:\Facets\Facets.exe"


def main() -> None:
    """Prepare, run, and process the full-run model reference-set decomposition."""

    parser = argparse.ArgumentParser(
        description="Run the full-run model reference-set severity decomposition experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Severity decomposition YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Processed bias-term CSV output path.",
    )
    parser.add_argument(
        "--vm-name",
        default=DEFAULT_VM_NAME,
        help="Parallels Windows VM name used to run FACETS.",
    )
    parser.add_argument(
        "--windows-project-path",
        default=DEFAULT_WINDOWS_PROJECT_PATH,
        help="Windows UNC path for the repository root.",
    )
    parser.add_argument(
        "--facets-exe",
        default=DEFAULT_FACETS_EXE,
        help="Windows path to Facets.exe.",
    )
    parser.add_argument(
        "--skip-facets",
        action="store_true",
        help="Only prepare and process existing FACETS output.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    print(f"config={config_path}")
    prep_outputs = run_severity_decomposition_facets(config_path)
    print(f"facets_data={prep_outputs.facets_data_path}")
    print(f"facets_spec={prep_outputs.facets_spec_path}")

    config = load_severity_decomposition_config(config_path)
    if not args.skip_facets:
        run_windows_facets(
            vm_name=args.vm_name,
            windows_project_path=args.windows_project_path,
            facets_exe=args.facets_exe,
            facets_run_dir=config.facets_run_dir,
            spec_filename=config.facets_spec_filename,
            output_filename=config.facets_output_filename,
        )

    postprocess_outputs = process_severity_decomposition_run(
        config_path=config_path,
        output_path=args.output.resolve(),
    )
    print(f"bias_terms={postprocess_outputs.bias_terms_path}")


def run_windows_facets(
    vm_name: str,
    windows_project_path: str,
    facets_exe: str,
    facets_run_dir: Path,
    spec_filename: str,
    output_filename: str,
) -> None:
    """Run FACETS in batch mode inside the configured Parallels Windows VM."""

    subprocess.run(["prlctl", "start", vm_name], check=False)
    relative_run_dir = facets_run_dir.resolve().relative_to(Path.cwd().resolve())
    windows_run_dir = windows_project_path + "\\" + str(relative_run_dir).replace("/", "\\")
    command = (
        f'pushd "{windows_run_dir}" && '
        f'start /w "" "{facets_exe}" "{spec_filename}" "{output_filename}" BATCH=YES && '
        "popd"
    )
    subprocess.run(["prlctl", "exec", vm_name, "cmd.exe", "/c", command], check=True)


if __name__ == "__main__":
    main()
