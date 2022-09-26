import os
from utils.synthetic_generator.generator import SyntheticGenerator

from Geosteering_model import *
from pyswarms.single.global_best import GlobalBestPSO





def interp(x, y, l):
    interpolation_model = interp1d(x, y, kind="quadratic")
    top_traj_x = np.linspace(x.min(), x.max(), l)
    top_traj_y = interpolation_model(top_traj_x)
    return top_traj_x, top_traj_y


def get_surf(points_real):
    points = points_real.copy()
    c = model.start_y 
    for i in range(len(points)):
        c += points[i]
        points[i] = c
    return points

def generate_sint_curve(points):
    x, y = [], []
    points = get_surf(points)


    top_traj_x, points = interp(x_coords, points, len(points)*5)
    
    
    top_y = list(points)
    bot_y = list(points + model.th)
    curve = generator.generate(top_traj_x, top_y, top_traj_x, bot_y).Points

    for point in curve: 
        x.append(point.Position)
        y.append(point.Value)

    df = pd.DataFrame([])
    df['MD'], df['Y'] = x, y
    df = df.query("MD >= @model.md_start")



    return interp(df.MD.to_numpy(), df.Y.to_numpy(), len(real_curve))


def fitness_function(X, x_coords, real_curve, model, generator):
    global best_cost, best_id 

    dist_arr = []
    for points in X:
        x, y = generate_sint_curve(points)
        d = dtaidistance.dtw.distance_fast(get_alpha(real_curve), get_alpha(y), window=50)
        dist_arr+=[d]

    dist_arr = np.array(dist_arr)
    if (d_min := dist_arr.min()) < best_cost:
        best_cost = d_min
        best_id = np.argmin(dist_arr)

    return dist_arr



if __name__ == '__main__':
    
    xml_path = 'input/well2withoutdips/well2withoutdips.xml'
    GR_path = 'input/well2withoutdips/GRwell2withoutdips.las'

    dim, increment_val = 10, 5
    n_particles = 100
    num_iter = 50


    best_cost, best_id = np.inf, None

    model = Geostering_Model(xml_path=xml_path,
                             GR_path=GR_path)

    model.start_y = get_val(model.intersection.x, model.markers_top, 'Y', 'X')
    real_curve = model.gamma_real.query("MD >= @model.md_start").value.to_numpy()



    scenario_path = os.getcwd() +'/'+ xml_path
    generator = SyntheticGenerator(scenario_path, model.gamma_offset.MD.to_list(), model.gamma_offset.value.to_list())


    x_coords = np.linspace(model.geosteering_part_tr.X.to_numpy()[0], model.geosteering_part_tr.X.to_numpy()[-1], dim)

    md_vals = [get_val(x, model.trajectory, 'MD', 'X') for x in x_coords]

    seism_y = np.array([get_val(x, model.markers_top, 'Y', 'X') for x in x_coords])
    seism_y_derivs = np.insert(np.diff(seism_y), 0, 0)

    
    # instatiate the optimizer
    x_max = increment_val * np.ones(dim)
    x_min = -increment_val * np.ones(dim)
    x_max[0], x_min[0] = 0.1, -0.1

    bounds = (x_min, x_max)
    options = {'c1': 1, 'c2': 1.3, 'w': 0.3}


    optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dim, options=options, bounds=bounds, velocity_clamp=(-1.5, 1.5))

    cost, pos = optimizer.optimize(fitness_function, num_iter, x_coords=x_coords, real_curve=real_curve, model=model, generator=generator)
    
    points = get_surf(pos)
    x_c, points = interp(x_coords, points, len(points)*5)

    top_y = list(points)
    bot_y = list(points + model.th)

    model.top_y, model.top_x = top_y, x_c
    model.bot_y, model.bot_x = bot_y, x_c
    model.save_results_to_xml(l=1)
    
    print('Done!')