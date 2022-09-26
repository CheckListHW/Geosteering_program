from geosteering_tools.python_scenario_xml_reader import *
from geosteering_tools.tools import *
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import dtaidistance
from scipy import stats
import lasio
from scipy.interpolate import interp1d, splrep, splev


class Intersection():
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y


class Geosteering_Model():
    def __init__(self, xml_path, GR_path):
        """Считываем все данные которые потребуются для геонавигации
        Args:
            xml_path (str): путь к xml файлу с данными GO
            GR_path (str): путь к las файлу с опорным каротажем
        """
        # считываем xml файл со сценарием
        self.Cor = Scenario()
        self.Cor.load(xml_path)

        # далее создаем датафреймы со всеми данными
        X, Y, MD = [], [], []
        for point in self.Cor.Trajectory.Points:
            X.append(point.SectionPoint.X)
            Y.append(point.SectionPoint.Y)
            MD.append(point.Md)
        trajectory = pd.DataFrame([])
        trajectory['MD'] = MD
        trajectory['X'] = X
        trajectory['Y'] = Y
        self.trajectory = trajectory.dropna()  # данные инклинометрии скважины

        MD, val = [], []
        for point in self.Cor.Property[0].Real.Points:
            val.append(point.Value)
            MD.append(point.Position)
        gamma_real = pd.DataFrame([])
        gamma_real['MD'] = MD
        gamma_real['value'] = val
        gamma_real = gamma_real[(np.abs(stats.zscore(gamma_real)) < 3).all(axis=1)]  # удаляем выбросы
        self.gamma_real = gamma_real.dropna()  # данные реального каротажа снятого со скважины при бурении
        clean_signal = filter_signal(self.gamma_real.value.to_numpy(), threshold=1e4)  # сглаживание/очистка от шумов
        self.gamma_real = self.gamma_real.iloc[:len(clean_signal)]
        self.gamma_real['value'] = get_alpha(clean_signal)

        X, Y = [], []
        for point in self.Cor.Section.Surfaces[0].Points:
            X.append(point.X)
            Y.append(point.Y)
        markers_top = pd.DataFrame([])
        markers_top['X'] = X
        markers_top['Y'] = Y
        self.markers_top = markers_top.dropna()  # изначальная поверхность кровли пласта

        X, Y = [], []
        for point in self.Cor.Section.Surfaces[1].Points:
            X.append(point.X)
            Y.append(point.Y)
        markers_bot = pd.DataFrame([])
        markers_bot['X'] = X
        markers_bot['Y'] = Y
        self.markers_bot = markers_bot.dropna()  # изначальная поверхность подошвы пласта

        # опорный каротаж считываем из las файла, а отметки кровля/подошва из xml
        MD, val = [], []
        for point in self.Cor.Property[0].Offset.Points:
            val.append(point.Value)
            MD.append(point.Position)
        gamma_offset = pd.DataFrame([])
        gamma_offset['MD'] = MD
        gamma_offset['value'] = val
        gamma_offset = gamma_offset.dropna()
        # self.a = gamma_offset.copy()

        top_offset_MD_ind = np.where(gamma_offset.MD.to_numpy() == 10000)[
            0]  # кровля пласта всегда лежит под MD == 10000
        bot_offset_MD_ind = np.where(gamma_offset.MD.to_numpy() == 20000)[
            0]  # подошва пласта всегда лежит под MD == 20000

        marker_ind = 0

        start_offset_point_ind = \
        gamma_offset.iloc[(gamma_offset['MD'] - self.Cor.Markers[marker_ind].OffsetPosition).abs().argsort()][
            'MD'].values[0]
        start_offset_point_ind = np.where(gamma_offset.MD.to_numpy() == start_offset_point_ind)[0]

        self.gamma_offset = lasio.read(GR_path).df().reset_index().rename(
            columns={'DEPT': 'MD', 'GR': 'value'})[['MD', 'value']]

        self.top_offset_MD = self.gamma_offset.MD.to_numpy()[top_offset_MD_ind][
            0]  # отсечка кровли пласта на опорном каротаже
        self.bot_offset_MD = self.gamma_offset.MD.to_numpy()[bot_offset_MD_ind][
            0]  # отсечка подошвы пласта на опорном каротаже

        self.start_offset_point = self.gamma_offset.MD.to_numpy()[start_offset_point_ind][
            0]  # проекция точки старта на опорный каротаж

        self.gamma_offset = self.gamma_offset[
            (np.abs(stats.zscore(self.gamma_offset)) < 3).all(axis=1)]  # удаляем выбросы
        self.gamma_offset.value = get_alpha(self.gamma_offset.value.to_numpy())

        # находим точку инициализации алгоритма путем нахождения точки пересечения траектории скважины с заданной кровлей пласта
        self.intersection = Intersection(self.Cor.Markers[marker_ind].Location.X,
                                         self.Cor.Markers[marker_ind].Location.Y)

        # if (self.top_offset_MD - self.start_offset_point) != 0:
        self.scale = (self.markers_bot.Y - self.markers_top.Y).mean() / (self.bot_offset_MD - self.top_offset_MD)
        # else:
        #   self.scale = 1
        self.th = (self.bot_offset_MD - self.top_offset_MD)  # обсчитываем мощность пласта
        self.th = self.th * self.scale

        # если пересечений траектории скважины с кровлей пласта несколько, берем только первое
        self.geosteering_part_tr = self.trajectory.query(
            "X >= @self.intersection.x")  # часть траектории где будет работать алгоритм

        self.md_start = get_val(self.intersection.x, self.trajectory, 'MD', target_column='X')
        md_end = self.geosteering_part_tr.MD.to_list()[-1]
        self.geosteering_part_len = md_end - self.md_start

        if self.Cor.Dips:
            # считываем интерпретированные геологом данные имиджа
            dips = [[d.Md, d.Dip, d.Location.X, d.Location.Y] for d in self.Cor.Dips]
            self.dips = pd.DataFrame(dips)
            self.dips.columns = ['MD', 'Dip', 'X', 'Y']
            self.dips = self.dips.sort_values(by='MD', ascending=True).query("X >= @self.intersection.x")
            # задаём параметры по умолчанию
            self.init_algoritm_params(self.md_start, md_end, 7, 35, 0.5, 'dtw')
            self.init_dips_algoritm_params(150)

        else:
            self.init_algoritm_params(self.md_start, md_end, 7, 35, 0.5, 'dtw')
            self.use_geosteering_alg = [True]
            self.control_points = [get_val(self.md_start, self.trajectory, 'X', 'MD'),
                                   get_val(md_end, self.trajectory, 'X', 'MD')]

    def init_increments_using_dips(self):
        deriv_vals = np.tan(self.dips.Dip)

        md_vals = []
        for md in self.dips.MD:
            md_vals.append(
                self.geosteering_part_tr.iloc[(self.geosteering_part_tr['MD'] - md).abs().argsort()]['MD'].values[0])

        derivs = []
        c = 0
        for md in self.geosteering_part_tr.MD:
            if md in md_vals:
                derivs.append(deriv_vals[c])
                c += 1
            else:
                derivs.append(np.nan)

        self.geosteering_part_tr['deriv'] = derivs

        deriv = self.geosteering_part_tr.deriv.interpolate(method='polynomial', order=0)

        deriv = deriv.fillna(deriv.dropna().to_numpy()[0]).to_numpy()
        deriv[0] = 0

        deriv = deriv * np.insert(np.diff(self.geosteering_part_tr.X.to_numpy()), 0, 0)

        self.geosteering_part_tr.drop(columns=['deriv'], inplace=True)
        self.geosteering_part_tr['increment'] = deriv

    def marker_using_geosteering_alg(self):
        self.control_points = [get_val(self.md_start, self.trajectory, 'X')] + self.dips.X.to_list() + [
            self.trajectory.X.to_list()[-1]]
        self.interpolation_segments_size = abs(np.diff(self.control_points))
        self.use_geosteering_alg = [i > self.min_allow_size for i in self.interpolation_segments_size]

    def init_dips_algoritm_params(self, min_allow_size):
        self.min_allow_size = min_allow_size
        self.marker_using_geosteering_alg()
        self.init_increments_using_dips()

    def init_algoritm_params(self, md_start, md_end, num_of_segments, delta_deg, st, metric, ang1=None, ang2=None,
                             incr=0):
        """Метод для установки гтперпараметров алгоритма

        Args:
            num_of_segments (int): колличество отрезков на которое следует разбить всю траекторию
            delta_deg (float): градус на который может отклоняться плоскость от горизонтали (условно при горизонтальном бурении)
            st (float): шаг для генерации синтетических кривых
            metric (str): метрика для сравнения кривых
        """
        self.ang1, self.ang2 = ang1, ang2
        self.incr = incr

        md = self.trajectory.query("MD >= @md_start & MD <= @md_end").MD.to_numpy()

        self.step = (
                                md.max() - md.min()) / num_of_segments  # длина отрезка на котором предположительно пласт распространяется линейно

        self.delta_deg = delta_deg  # максимальный угол между начальной и конечной точкой отрезка
        # (условно, при горизонтальном расположении траектории скважины)
        # по сути параметр для пересчета глубины на которую может сместиться проекция
        # скважины на опорном каротаже

        self.st = st  # шаг в метрах с которым генерируется синтетика, по сути контроль точности предсказания/скорости алгоритма

        if metric == 'mse':
            self.dist_calculation = lambda curve1, curve2: mean_squared_error(curve1, curve2)
            self.get_best_sintetic_curve_ind = lambda d_l: np.argmin(d_l)
        elif metric == 'mae':
            self.dist_calculation = lambda curve1, curve2: mean_absolute_error(curve1, curve2)
            self.get_best_sintetic_curve_ind = lambda d_l: np.argmin(d_l)
        elif metric == 'r2':
            self.dist_calculation = lambda curve1, curve2: r2_score(curve1, curve2)
            self.get_best_sintetic_curve_ind = lambda d_l: np.argmax(d_l)
        elif metric == 'dtw':
            self.dist_calculation = lambda curve1, curve2: dtaidistance.dtw.distance(curve1, curve2, window=10)
            self.get_best_sintetic_curve_ind = lambda d_l: np.argmin(d_l)
        # elif metric == 'cos_sim':
        #     self.dist_calculation = lambda curve1, curve2: cos_sim(curve1, curve2)
        #     self.get_best_sintetic_curve_ind = lambda d_l: np.argmax(d_l)
        # elif metric == 'lp_distance':
        #     self.dist_calculation = lambda curve1, curve2: lp_distance(curve1, curve2)
        #     self.get_best_sintetic_curve_ind = lambda d_l: np.argmin(d_l)

        # рассчитываем количество отрезков по которым будем матчить кривые
        self.num_of_iterations = num_of_segments

    def sint_curve_generator(self, start_offset_point, gamma_real_cut, delta, d):
        """Метод для генерации синтетической кривой в соответствии с длиной реальной кривой

        Args:
            start_offset_point (float): начальная точка из которой будут генерироваться кривые
            gamma_real_cut (Pandas.DataFrame): датафрейм содержащий реальную кривую
            delta (float): максимальное отклонение проекции скважины на опорный каротаж от начальной точки в метрах 

        Returns:
            List: список с синтетическими каротажами 
            List: список конечных точек проекии для синтетических каротажей
            float: Y координата конца отрезка

        """
        real_tr = [get_val(jtem, self.trajectory, 'Y') for jtem in gamma_real_cut.MD]
        curvature = (real_tr - np.linspace(real_tr[0], real_tr[-1],
                                           len(real_tr))) / self.scale  # рассчитываем кривизну скважины на заданном участке
        end_points = np.arange(start_offset_point + d - delta, start_offset_point + d + delta + self.st, self.st)
        sintetic_curves = [np.linspace(start_offset_point,
                                       end_point,
                                       len(gamma_real_cut.MD)) + curvature for end_point in
                           end_points]  # рассчитываем проекции получившейся траектории на опорный каротаж

        # снимаем значения с опорного каротажа и формируем итоговые кривые
        for i, item in enumerate(sintetic_curves):
            for j, jtem in enumerate(item):
                try:
                    gamma_val = get_val(jtem, self.gamma_offset, 'value')
                except IndexError:
                    gamma_val = \
                    self.gamma_offset.iloc[(self.gamma_offset['MD'] - jtem).abs().argsort()]['value'].values[0]
                sintetic_curves[i][j] = gamma_val

        return sintetic_curves, end_points, real_tr[-1]

    def geosteering_iteration_using_log(self, delt, start_offset_point, md_start, ang=None, plot_matching=False):
        """Итерация процесса геонавигации, на которой матчится один отрезок

        Args:
            start_offset_point (float): проекция стартовой точки на опорном каротаже
            md_start (float): отметка глубины по скважине для начала геонавигации
            plot_matching (bool, optional): Визуализация кривых которые были выбраны по заданной метрике как наиболее похожие. Defaults to False.

        Returns:
            Tuple: кортеж с проекциями на опроный каротаж, 1 значение - начало отрезка, 2 - конец  
            Tuple: кортеж с отметками глубин по скважине, 1 значение - начало отрезка, 2 - конец 
            List: синтетическая кривая по метрике наилучшим образом сходящаяся с реальной
            float: Х координата конца отрезка
            float: Y координата конца отрезка
        """
        md_next = md_start + self.step
        gamma_real_cut = self.gamma_real.query("MD <= @md_next & MD >= @md_start")
        tr_x = get_val(gamma_real_cut.MD.to_numpy()[-1], self.trajectory, 'X')

        # пересчет из угла в метры (даны угол между катетом и гипотенузой, рассчитывается катет напротив угла)
        if not ang:
            delta = (tr_x - get_val(gamma_real_cut.MD.to_numpy()[0], self.trajectory, 'X')) * np.tan(
                np.radians(self.delta_deg))
            d = 0
        else:
            delta = delt
            d = (tr_x - get_val(gamma_real_cut.MD.to_numpy()[0], self.trajectory, 'X')) * np.tan(ang * -1)

        # генерируем синтетические кривые
        sintetic_curves, end_points, tr_y = self.sint_curve_generator(start_offset_point, gamma_real_cut, delta, d)

        # рассчитываем метрику схожести между реальным каротажом и синтетическими
        dist_list = []
        real_curve = gamma_real_cut.value.to_numpy()
        for curve in sintetic_curves:
            dist_list.append(self.dist_calculation(real_curve, curve))

        # находим наилучший результат по заданной метрике и если нужно визуализируем её
        best_sintetic_curve_ind = self.get_best_sintetic_curve_ind(dist_list)
        # if plot_matching:
        #     plot_curves(sintetic_curves[best_sintetic_curve_ind], gamma_real_cut.value)

        offset_md_points = start_offset_point, end_points[best_sintetic_curve_ind]
        trajectory_md_points = md_start, md_next

        return offset_md_points, trajectory_md_points, sintetic_curves[best_sintetic_curve_ind], tr_x, tr_y

    def start_geosteering(self, start_offset_point, md_start, plot_matching=False):
        """Метод осуществляющий геонавигацию. Итеративно бежит и выбирает положение отрезка в пространстве жадным образом, то есть лучшее решение на каждом шаге.

        Args:
            plot_matching (bool, optional): Визуализация кривых которые были выбраны по заданной метрике как наиболее похожие. Defaults to False.
        """

        top_traj_y = []
        top_traj_x = []

        self.sintetic_curve = []
        increment = self.incr
        for it in range(self.num_of_iterations):
            if it == 0:
                ang = self.ang1
                delt = 5
            elif it == self.num_of_iterations - 1:
                ang = self.ang2
                delt = 5
            else:
                ang = None

            offset_md_points, trajectory_md_points, s_c, tr_x, tr_y = self.geosteering_iteration_using_log(
                md_start=md_start,
                start_offset_point=start_offset_point,
                plot_matching=plot_matching,
                ang=ang, delt=delt)

            self.sintetic_curve += list(s_c)

            increment += (offset_md_points[1] - offset_md_points[0]) * self.scale

            top_traj_y.append(tr_y - increment)
            top_traj_x.append(tr_x)

            md_start = trajectory_md_points[1]
            start_offset_point = offset_md_points[1]

        return top_traj_x, top_traj_y

    def start_complex_geosteering(self, min_interp_seg_size, delta_deg, st, metric, plot_matching, without_dips_nof=7):
        if self.Cor.Dips:
            self.init_dips_algoritm_params(min_interp_seg_size)
            dips = [None] + self.dips.Dip.to_list() + [None]
        else:
            dips = [None, None]

        start_point = (
        self.intersection.x, self.intersection.y + (self.top_offset_MD - self.start_offset_point) * self.scale)
        top_traj_y, top_traj_x = [start_point[1]], [start_point[0]]

        for i, item in enumerate(self.use_geosteering_alg):
            if item:
                ang1, ang2 = dips[i], dips[i + 1]

                s_p, e_p = self.control_points[i], self.control_points[i + 1]
                start_offset_point, incr = self.get_proj(start_point[0], start_point[1])
                # print(start_offset_point, self.start_offset_point)

                md_start = get_val(s_p, self.trajectory, 'MD', 'X')
                md_end = get_val(e_p, self.trajectory, 'MD', 'X')

                if int((e_p - s_p)) == int(self.geosteering_part_len):
                    num_of_segments = without_dips_nof
                elif (e_p - s_p) >= 150:
                    num_of_segments = 2
                elif 90 > (e_p - s_p) > 150:
                    num_of_segments = 1
                # elif (e_p - s_p) > 80:
                #     num_of_segments = 2
                else:
                    num_of_segments = 1

                self.init_algoritm_params(md_start, md_end, num_of_segments, delta_deg, st, metric, ang1, ang2,
                                          incr=incr)  # настраиваем гиперпараметры алгоритма
                x, y = self.start_geosteering(start_offset_point, md_start, plot_matching=plot_matching)

                top_traj_y += list(y)
                top_traj_x += list(x)

                start_point = (top_traj_x[-1], top_traj_y[-1])


            else:
                s_p, e_p = self.control_points[i], self.control_points[i + 1]
                cut_df = self.geosteering_part_tr.query("X >= @s_p & X <= @e_p")
                deriv = cut_df.increment.to_numpy()
                deriv[0] = 0
                for i in range(len(deriv[1:])):
                    deriv[i + 1] = deriv[i] + deriv[i + 1]

                y = list(deriv + start_point[1])
                x = cut_df.X.to_list()
                top_traj_y += y
                top_traj_x += x
                start_point = (top_traj_x[-1], top_traj_y[-1])

        self.center_traj_x_without_smoothing = np.array(top_traj_x)
        self.center_traj_y_without_smoothing = np.array(top_traj_y + self.th / 2)

        top_traj_x = np.array(top_traj_x)
        top_traj_y = np.array(top_traj_y + self.th / 2)

        interpolation_model = interp1d(top_traj_x, top_traj_y, kind="quadratic")

        top_traj_x = np.linspace(top_traj_x.min(), top_traj_x.max(), len(top_traj_x) * 5)
        top_traj_y = interpolation_model(top_traj_x)
        top_traj_x, top_traj_y = self.present_interp_top(top_traj_x, top_traj_y)

        res = parallel_curves(top_traj_x, top_traj_y, d=self.th * 0.5, flag1=False)
        self.top_y, self.top_x = res['y_outer'], res['x_outer']
        self.bot_y, self.bot_x = res['y_inner'], res['x_inner']

    def save_results_to_xml(self, l=150, path = 'output.xml'):
        """
        Сохранение результатов в xml
        """
        new_top_markers = []
        new_bot_markers = []

        for x_top, y_top in zip(self.top_x[::200], self.top_y[::200]):
            new_top_markers.append(Point(x=x_top, y=y_top))

        for x_bot, y_bot in zip(self.bot_x[::200], self.bot_y[::200]):
            new_bot_markers.append(Point(x=x_bot, y=y_bot))

        top_surface_points = []
        for point in self.Cor.Section.Surfaces[0].Points:
            if (point.X + 5) < self.center_traj_x_without_smoothing[0]:
                top_surface_points.append(point)
            else:
                break

        bot_surface_points = []
        for point in self.Cor.Section.Surfaces[1].Points:
            if (point.X + 5) < self.center_traj_x_without_smoothing[0]:
                bot_surface_points.append(point)
            else:
                break

        new_top_markers = top_surface_points[:-1] + new_top_markers
        new_bot_markers = bot_surface_points[:-1] + new_bot_markers

        self.Cor.Section.Surfaces[0].Points = new_top_markers
        self.Cor.Section.Surfaces[1].Points = new_bot_markers

        self.Cor.save_xml(path)

    def get_proj(self, x, y):
        """
        Метод для получения значения проекции скважины на опорный 
        каротаж из знания положения точки кровли пласта 

        Args:
            x (float): х координата кровли
            y (float): у координата кровли

        Returns:
            float: значение проекции (глубина на опорном каротаже)
        """
        well_y = get_val(x, self.trajectory, 'Y', 'X')
        # print(well_y, y)
        delta = well_y - y
        # print(delta)
        return self.top_offset_MD + delta / self.scale, delta

    def present_interp_top(self, top_x, top_y, d=100):
        top_x_y = pd.DataFrame(columns=['X', 'Y'])
        top_x_sorting = top_x.copy()
        top_x_sorting.sort()
        top_x_y['X'] = top_x_sorting
        top_x_y['Y'] = top_y
        markers_top = self.markers_top[['X', 'Y']][self.markers_top['X'] > max(top_x_y['X']) + d] + self.th * 0.5
        top_x_y = top_x_y.append(markers_top)

        tck = splrep(top_x_y['X'], top_x_y['Y'])
        top_x = np.arange(top_x_y['X'].min(), top_x_y['X'].max())
        top_y = splev(top_x, tck)
        return top_x, top_y

    # def present_interp_bot(self, offset=100):
    #     bot_x_y = pd.DataFrame(columns=['X', 'Y'])
    #     bott_x_sorting = self.bot_x.copy()
    #     bott_x_sorting.sort()
    #     bot_x_y['X'] = bott_x_sorting
    #     # bot_x_y['X'] = self.bot_x
    #     bot_x_y['Y'] = self.bot_y
    #     markers_bot = self.markers_bot[['X', 'Y']][self.markers_bot['X']>max(bot_x_y['X'])+offset]
    #     bot_x_y = bot_x_y.append(markers_bot)
    #     tck = splrep(bot_x_y['X'], bot_x_y['Y'])
    #     self.bot_x = np.arange(bot_x_y['X'].min(),
    #                            bot_x_y['X'].max())
    #     self.bot_y = splev(self.bot_x, tck)
