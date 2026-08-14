import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

sys_path = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from upscale_engine import (  # noqa: E402
    ProgressReporter,
    UpscaleEngine,
    resolve_workflow_params,
)


class _CliReporter(ProgressReporter):
    """Emits the legacy stdout protocol parsed by the C# GUI."""

    def log(self, message: str) -> None:
        print(message, flush=True)

    def archive_total(self, total: int) -> None:
        print(f"TOTALZIP={total}", flush=True)

    def file_completed(self, kind: str) -> None:
        print(f"PROGRESS={kind}", flush=True)


def parse_settings_from_cli():
    parser = argparse.ArgumentParser(
        prog="python run_upscale.py",
        description=(
            "By default, used by MangaJaNaiConverterGui as an internal tool. "
            "Alternative options made available to make it easier to skip the GUI "
            "and run upscaling jobs directly from CLI."
        ),
    )

    execution_type_group = parser.add_mutually_exclusive_group(required=True)
    execution_type_group.add_argument(
        "--settings",
        help="Default behaviour, based on provided appstate configuration. For advanced usage.",
    )
    execution_type_group.add_argument(
        "-f", "--file-path", help="Upscale single file"
    )
    execution_type_group.add_argument(
        "-d", "--folder-path", help="Upscale whole directory"
    )

    parser.add_argument(
        "-o",
        "--output-folder-path",
        default=os.path.join(".", "out"),
        help="Output directory for upscaled files. Default: ./out",
    )
    parser.add_argument(
        "-m",
        "--models-directory-path",
        default=os.path.join("..", "models"),
        help=(
            "Directory with models used for upscaling. "
            "Supports only models bundled with MangaJaNaiConvertedGui. "
            "Default: MangaJaNaiConverterGui/chaiNNer/models/"
        ),
    )
    parser.add_argument(
        "-u",
        "--upscale-factor",
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help="Used for calculating which model will be used. Default: 2",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help=(
            "Device used to run upscaling jobs in case more than one is available. "
            "Default: 0"
        ),
    )

    args = parser.parse_args()

    return parse_auto_settings(args) if args.settings else parse_manual_settings(args)


def parse_auto_settings(args):
    with open(args.settings, encoding="utf-8") as f:
        json_settings = json.load(f)

    return json_settings


def parse_manual_settings(args):
    default_file_path = os.path.join("..", "resources", "default_cli_configuration.json")
    with open(default_file_path, "r") as default_file:
        default_json = json.load(default_file)

    default_json["SelectedDeviceIndex"] = int(args.device_index)
    default_json["ModelsDirectory"] = args.models_directory_path

    default_json["Workflows"]["$values"][0]["OutputFolderPath"] = args.output_folder_path
    default_json["Workflows"]["$values"][0]["SelectedDeviceIndex"] = args.device_index
    default_json["Workflows"]["$values"][0]["UpscaleScaleFactor"] = args.upscale_factor
    if args.file_path:
        default_json["Workflows"]["$values"][0]["SelectedTabIndex"] = 0
        default_json["Workflows"]["$values"][0]["InputFilePath"] = args.file_path
    elif args.folder_path:
        default_json["Workflows"]["$values"][0]["SelectedTabIndex"] = 1
        default_json["Workflows"]["$values"][0]["InputFolderPath"] = args.folder_path

    return default_json


def main() -> None:
    settings = parse_settings_from_cli()
    workflow = settings["Workflows"]["$values"][settings["SelectedWorkflowIndex"]]

    engine = UpscaleEngine(settings, reporter=_CliReporter())

    print(
        "settings",
        engine.settings_parser.get_int("accelerator_device_index", 0),
        flush=True,
    )

    (
        image_format,
        target_scale,
        target_width,
        target_height,
        grayscale_detection_threshold,
    ) = resolve_workflow_params(workflow)

    start_time = time.time()

    if workflow["SelectedTabIndex"] == 1:
        engine.upscale_folder(
            workflow["InputFolderPath"],
            workflow["OutputFolderPath"],
            workflow["OutputFilename"],
            workflow["UpscaleImages"],
            workflow["UpscaleArchives"],
            workflow["OverwriteExistingFiles"],
            image_format,
            workflow["LossyCompressionQuality"],
            workflow["UseLosslessCompression"],
            target_scale,
            target_width,
            target_height,
            workflow["Chains"]["$values"],
            grayscale_detection_threshold,
        )
    elif workflow["SelectedTabIndex"] == 0:
        engine.upscale_file(
            workflow["InputFilePath"],
            workflow["OutputFolderPath"],
            workflow["OutputFilename"],
            workflow["OverwriteExistingFiles"],
            image_format,
            workflow["LossyCompressionQuality"],
            workflow["UseLosslessCompression"],
            target_scale,
            target_width,
            target_height,
            workflow["Chains"]["$values"],
            grayscale_detection_threshold,
        )

    elapsed_time = time.time() - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
