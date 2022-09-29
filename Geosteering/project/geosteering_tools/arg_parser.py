import argparse
import os


def arg_parser() -> (argparse.Namespace, [str]):
    parser = argparse.ArgumentParser(description='Geosteering')
    parser.add_argument(
        '--scenario_path',
        type=str,
        default='input.xml',
        help='Путь до сценария(scenario)  в формате .xml (default: "input.xml")'
    )

    parser.add_argument(
        '--offset_path',
        type=str,
        default='input.las',
        help='Путь до файла с каратажом опорных скважин в формате .las  (default: "input.las")'
    )

    parser.add_argument(
        '--result_path',
        type=str,
        default='files.xml',
        help='Путь сохранения файла (default: output.xml)'
    )

    return parser.parse_args()
