import argparse
import os


def arg_parser() -> (argparse.Namespace, [str]):
    metrics = ['lp_distance', 'cos_sim', 'dtw', 'r2', 'mse', 'mae']
    parser = argparse.ArgumentParser(description='Geosteering')
    parser.add_argument(
        '--scenario_path',
        type=str,
        default='input/input.xml',
        help='Путь до сценария(scenario)  в формате юxml (default: "input/input.xml")'
    )

    parser.add_argument(
        '--gr_path',
        type=str,
        default='input/input.las',
        help='Путь до файла с каратажом опорных скважин в формате .las  (default: "input/input.las")'
    )

    parser.add_argument(
        '--segments_count',
        type=int,
        default=5,
        help='Количество отрезков на которую разделится траектория скважины (default: 5)'
    )

    parser.add_argument(
        '--delta_deg',
        type=int,
        default=30,
        help='Максимально допустимый угол отклонения от горизонтали в интервале (0;90) (default: 30)'
    )

    parser.add_argument(
        '--st',
        type=int,
        default=1,
        help='Частота генерации поверхности (в метрах).  (default: 1)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='lp_distance',
        help=f'Метод сравнения схожести кривых. Доступны: {", ".join(metrics)} (default: lp_distance)'
    )

    parser.add_argument(
        '--result_path',
        type=str,
        default='output.xml',
        help='Путь сохранения файла (default: output/output.xml)'
    )

    current_namespace = parser.parse_args()

    error_messages = []
    if not os.path.exists(current_namespace.scenario_path):
        error_messages.append(f'file {current_namespace.scenario_path} is not exist')

    if not os.path.exists(current_namespace.gr_path):
        error_messages.append(f'file {current_namespace.gr_path} is not exist')

    if not 0 < current_namespace.delta_deg < 90:
        error_messages.append('delta_deg should be in (0:90)')

    if current_namespace.segments_count < 1:
        error_messages.append('segments_count should be more than 0')

    if current_namespace.st < 1:
        error_messages.append('st should be more than 0')

    if current_namespace.metric not in metrics:
        error_messages.append(f'metric {metrics} must be one of: {", ".join(metrics)}')

    open(current_namespace.result_path, 'w+').close()
    return current_namespace, error_messages
