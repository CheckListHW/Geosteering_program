from geosteering_tools.python_scenario_xml_reader import *
from geosteering_tools.tools import *
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import dtaidistance
from shapely import geometry
from shapely.geometry import LineString
from scipy import stats
import lasio
from scipy.interpolate import interp1d


class Geostering_Model():    
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

        top_offset_MD_ind = np.where(gamma_offset.MD.to_numpy() == 10000)[0]  # кровля пласта всегда лежит под MD == 10000
        bot_offset_MD_ind = np.where(gamma_offset.MD.to_numpy() == 20000)[0]  # подошва пласта всегда лежит под MD == 20000

        self.gamma_offset = lasio.read(GR_path).df().reset_index().rename(columns={'DEPT': 'MD', 'GR': 'value'})[
            ['MD', 'value']]

        self.top_offset_MD = self.gamma_offset.MD.to_numpy()[top_offset_MD_ind][0]  # отсечка кровли пласта на опорном каротаже
        self.bot_offset_MD = self.gamma_offset.MD.to_numpy()[bot_offset_MD_ind][0]  # отсечка подошвы пласта на опорном каротаже
        self.th = self.bot_offset_MD - self.top_offset_MD  # обсчитываем мощность пласта

        self.gamma_offset = self.gamma_offset[(np.abs(stats.zscore(self.gamma_offset)) < 3).all(axis=1)]  # удаляем выбросы
        self.gamma_offset.value = get_alpha(self.gamma_offset.value.to_numpy())


        # находим точку инициализации алгоритма путем нахождения точки пересечения траектории скважины с заданной кровлей пласта
        tr_x = trajectory.X.to_numpy()
        tr_y = trajectory.Y.to_numpy()
        markers_top_x = markers_top.X.to_numpy()
        markers_top_y = markers_top.Y.to_numpy()
        first_line = LineString(np.column_stack((tr_x, tr_y)))
        second_line = LineString(np.column_stack((markers_top_x, markers_top_y)))
        self.intersection = first_line.intersection(second_line)
        # если пересечений траектории скважины с кровлей пласта несколько, берем только первое
        if isinstance(self.intersection, geometry.multipoint.MultiPoint):
            self.intersection = self.intersection[0]

        # считываем интерпретированные геологом данные имиджа
        dips = [[d.Md, d.Dip, d.Location.X, d.Location.Y ] for d in self.Cor.Dips]
        self.dips = pd.DataFrame(dips)
        self.dips.columns = ['MD', 'Dip', 'X', 'Y']

        # задаём параметры по умолчанию
        self.init_algoritm_params(7, 35, 0.5, 'dtw')


    def init_algoritm_params(self, num_of_segments, delta_deg, st, metric):
        """Метод для установки гтперпараметров алгоритма

        Args:
            num_of_segments (int): колличество отрезков на которое следует разбить всю траекторию
            delta_deg (float): градус на который может отклоняться плоскость от горизонтали (условно при горизонтальном бурении)
            st (float): шаг для генерации синтетических кривых
            metric (str): метрика для сравнения кривых
        """        
        self.md_start = get_val(self.intersection.x, self.trajectory, 'MD', target_column='X')
        md = self.trajectory.query("MD >= @self.md_start").MD.to_numpy()

        self.step = (md.max() - md.min())/num_of_segments  # длина отрезка на котором предположительно пласт распространяется линейно

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
        elif metric == 'cos_sim':
            self.dist_calculation = lambda curve1, curve2: cos_sim(curve1, curve2)
            self.get_best_sintetic_curve_ind = lambda d_l: np.argmax(d_l)
        elif metric == 'lp_distance':
            self.dist_calculation = lambda curve1, curve2: lp_distance(curve1, curve2)
            self.get_best_sintetic_curve_ind = lambda d_l: np.argmin(d_l)

        # рассчитываем количество отрезков по которым будем матчить кривые
        self.num_of_iterations = num_of_segments   #self.calculate_num_of_iterations()


    # def calculate_num_of_iterations(self):
    #     top_offset_MD = self.top_offset_MD
    #     md = self.gamma_real.query("MD >= @top_offset_MD").MD.to_numpy()
    #     num = (md.max() - md.min()) / self.step
    #     # если последний отрезок меньше заданного значения (пока задал как 30 метров), то его не матчим
    #     if (md.max() - md.min()) % self.step >= 30:
    #         return int(num) + 1
    #     else:
    #         return int(num)


    def sint_curve_generator(self, start_offset_point, gamma_real_cut, delta):
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
        curvature = real_tr - np.linspace(real_tr[0], real_tr[-1], len(real_tr)) # рассчитываем кривизну скважины на заданном участке

        end_points = np.arange(start_offset_point - delta, start_offset_point + delta + self.st, self.st)
        sintetic_curves = [np.linspace(start_offset_point, end_point, len(gamma_real_cut.MD)) + curvature for end_point
                           in end_points]   # рассчитываем проекции получившейся траектории на опорный каротаж

        # снимаем значения с опорного каротажа и формируем итоговые кривые
        for i, item in enumerate(sintetic_curves):
            for j, jtem in enumerate(item):
                try:
                    gamma_val = get_val(jtem, self.gamma_offset, 'value')
                except IndexError:
                    gamma_val = self.gamma_offset.iloc[(self.gamma_offset['MD'] - jtem).abs().argsort()]['value'].values[0]
                sintetic_curves[i][j] = gamma_val

        return sintetic_curves, end_points, real_tr[-1]


    def geosteering_iteration(self,start_offset_point, md_start, plot_matching=False):
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
        delta = (tr_x - get_val(gamma_real_cut.MD.to_numpy()[0], self.trajectory, 'X')) * np.tan(np.radians(self.delta_deg))

        # генерируем синтетические кривые
        sintetic_curves, end_points, tr_y = self.sint_curve_generator(start_offset_point, gamma_real_cut, delta)

        # рассчитываем метрику схожести между реальным каротажом и синтетическими
        dist_list = []
        real_curve = gamma_real_cut.value.to_numpy()
        for curve in sintetic_curves:
            dist_list.append(self.dist_calculation(real_curve, curve))

        # находим наилучший результат по заданной метрике и если нужно визуализируем её
        best_sintetic_curve_ind = self.get_best_sintetic_curve_ind(dist_list)
        if plot_matching:
            plot_curves(sintetic_curves[best_sintetic_curve_ind], gamma_real_cut.value)

        offset_md_points = start_offset_point, end_points[best_sintetic_curve_ind]
        trajectory_md_points = md_start, md_next

        return offset_md_points, trajectory_md_points, sintetic_curves[best_sintetic_curve_ind], tr_x, tr_y


    def start_geosteering(self, plot_matching=False):
        """Метод осуществляющий геонавигации. Итеративно бежит и выбирает положение отрезка в пространстве жадным образом, то есть лучшее решение на каждом шаге.

        Args:
            plot_matching (bool, optional): Визуализация кривых которые были выбраны по заданной метрике как наиболее похожие. Defaults to False.
        """        
        start_offset_point = self.top_offset_MD

        md_start = self.md_start

        top_traj_y = [get_val(self.intersection.x, self.trajectory, 'Y', target_column='X')]

        top_traj_x = [get_val(self.intersection.x, self.trajectory, 'X', target_column='X')]

        # start_point = trajectory.iloc[(trajectory['X'] - intersection.x).abs().argsort()]
        # md_start = start_point.MD.values[0]
        # top_traj_y = [start_point.Y.values[0]]
        # top_traj_x = [start_point.X.values[0]]

        offset_md_points_list = []
        trajectory_md_points_list = []
        self.sintetic_curve = []
        increment = 0
        for iter in range(self.num_of_iterations):
            offset_md_points, trajectory_md_points, s_c, tr_x, tr_y = self.geosteering_iteration(md_start=md_start,
                                                                                                 start_offset_point=start_offset_point,
                                                                                                 plot_matching=plot_matching)

            self.sintetic_curve += list(s_c)
            offset_md_points_list.append(offset_md_points)
            trajectory_md_points_list.append(trajectory_md_points)
            increment += offset_md_points[1] - offset_md_points[0]

            top_traj_y.append(tr_y - increment)
            top_traj_x.append(tr_x)

            md_start = trajectory_md_points[1]
            start_offset_point = offset_md_points[1]

        self.center_traj_x_without_smoothing = np.array(top_traj_x)
        self.center_traj_y_without_smoothing = np.array(top_traj_y + self.th*0.5)

        top_traj_x = np.array(top_traj_x)
        top_traj_y = np.array(top_traj_y + self.th*0.5)

        cubic_interpolation_model = interp1d(top_traj_x, top_traj_y, kind="cubic")

        top_traj_x = np.linspace(top_traj_x.min(), top_traj_x.max(), len(top_traj_x) * 5)
        top_traj_y = cubic_interpolation_model(top_traj_x)

        res = parallel_curves(top_traj_x, top_traj_y, d=self.th*0.5, flag1=False)
        self.top_y, self.top_x = res['y_outer'], res['x_outer']
        self.bot_y, self.bot_x = res['y_inner'], res['x_inner']


    def surfaces_visualization(self, d_f=5, n=30000):
        """Визуализация итоговых поверхностей

        Args:
            d_f (int, optional): Сколько метров выше и ниже кроавли и подошвы пласта отобразить на опорном каротаже. Defaults to 5.
            n (int, optional): Сколько точек от устья скважины не отображать. Defaults to 30000.
        """        
        g_offset = self.gamma_offset.copy()
        scaler = MinMaxScaler(feature_range=(0, 50))
        g_offset_value = g_offset.value.to_numpy()
        g_offset.value = scaler.fit_transform(g_offset_value.reshape(g_offset_value.shape[0], 1))
        g_offset = g_offset.query("MD >= @self.top_offset_MD-@d_f  & MD <= @self.bot_offset_MD+@d_f ")

        fig = px.line(x=self.trajectory.X[n:], y=self.trajectory.Y[n:])

        fig.add_scatter(x=self.markers_top.X, y=self.markers_top.Y, marker=dict(color='green'))
        fig.add_scatter(x=self.markers_bot.X, y=self.markers_bot.Y, marker=dict(color='green'))

        fig.add_scatter(x=self.top_x,
                        y=self.top_y,
                        marker=dict(color='red'))

        fig.add_scatter(x=self.bot_x,
                        y=self.bot_y,
                        marker=dict(color='red'))

        fig.add_scatter(x=self.center_traj_x_without_smoothing,
                        y=self.center_traj_y_without_smoothing,
                        marker=dict(color='black'))

        for point, d in zip(self.center_traj_x_without_smoothing, self.center_traj_y_without_smoothing):
            x_val = point + g_offset.value.to_numpy() - g_offset.value.mean()
            y_val = []
            a = d
            for diff in g_offset.MD.diff().fillna(0).to_numpy():
                a += diff
                y_val.append(a - d_f)

            fig.add_scatter(x=x_val,
                            y=y_val,
                            marker=dict(color='orange'))

        fig.update_yaxes(autorange="reversed")
        fig.show()


    def curve_matching_visualization(self):
        """Визуализация для сравнения реального и итогового синтетического каротажа
        """        
        plt.figure(figsize=(25, 7))
        plot_curves(self.sintetic_curve,
                    self.gamma_real[self.gamma_real.MD >= self.md_start].value.to_numpy()[:len(self.sintetic_curve)])


    def save_results_to_xml(self):
        new_top_markers = []
        new_bot_markers = []

        for x_top, y_top in zip(self.top_x, self.top_y):
            new_top_markers.append(Point(x=x_top, y=y_top))

        for x_bot, y_bot in zip(self.bot_x, self.bot_y):
            new_bot_markers.append(Point(x=x_bot, y=y_bot))

        top_surface_points = []
        for point in self.Cor.Section.Surfaces[0].Points:
            if point.X < self.center_traj_x_without_smoothing[0]:
                top_surface_points.append(point)
            else:
                break

        bot_surface_points = []
        for point in self.Cor.Section.Surfaces[1].Points:
            if point.X < self.center_traj_x_without_smoothing[0]:
                bot_surface_points.append(point)
            else:
                break

        new_top_markers = top_surface_points[:-1] + new_top_markers
        new_bot_markers = bot_surface_points[:-1] + new_bot_markers

        self.Cor.Section.Surfaces[0].Points = new_top_markers
        self.Cor.Section.Surfaces[1].Points = new_bot_markers

        self.Cor.save_xml('output/result.xml')

    


