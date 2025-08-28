"""Calculates the impact of SWV"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.show()
from scipy.interpolate import griddata
from ambiance import Atmosphere


molar_mass_air = 28.97 * 10 ** -3  # kg/mol
molar_mass_ch4 = 16.04 * 10 ** -3  # kg/mol
M_h2o = 18.01528 * 10 ** -3 # kg/mol
N_avogrado = 6.02214076*10**23


#MYHRE

def get_volume_matrix(heights, latitudes, delta_h, delta_deg):
    R = 6371000.  # Earth radius in meters
    # delta_h = 100.  # height increment in meters
    # delta_deg = 1.  # latitude increment
    delta_phi = np.deg2rad(delta_deg)

    # heights = np.arange(0, 60000 + delta_h, delta_h)  # 0 to 60 km
    # latitudes = np.arange(-85, 86, delta_deg)  # -85° to 85°

    # Volume of 1° latitude x 100 m height strip integrated over all longitude
    volumes = np.zeros((len(heights), len(latitudes)))

    for i, h in enumerate(heights):
        for j, lat in enumerate(latitudes):
            volumes[i, j] = 2 * np.pi * (R + h) ** 2 * np.cos(np.deg2rad(lat)) * delta_phi * delta_h

    # print("Volume shape:", volumes.shape)
    # print("Volume at equator, sea level:", volumes[0, 0])
    # print(volumes)
    return volumes

def construct_myhre_2b_df():
    # different lines for all lines in the plot
    b02 = [
        [-86.01027, 12.99162],
        [-82.06041, 13.10290],
        [-77.77865, 13.37418],
        [-74.25444, 13.68127],
        [-70.63141, 13.94329],
        [-67.00839, 14.26925],
        [-63.38537, 14.61653],
        [-59.76234, 15.07748],
        [-56.13932, 15.55974],
        [-52.51629, 15.86439],
        [-48.89327, 16.10509],
        [-45.27024, 16.36711],
        [-41.64722, 16.62912],
        [-38.02420, 16.90535],
        [-34.40117, 17.29525],
        [-30.77815, 17.75620],
        [-27.05631, 18.37530],
        [-22.98315, 19.75268],
        [-19.25034, 21.04298],
        [-15.79200, 22.62439],
        [-12.33366, 24.08928],
        [-8.71063, 25.14703],
        [-5.08761, 25.71456],
        [-1.46459, 25.96947],
        [2.15844, 25.84783],
        [5.78146, 25.31411],
        [9.40449, 24.50331],
        [13.02751, 23.38700],
        [16.65054, 21.90124],
        [20.27356, 20.51045],
        [23.09147, 19.47935],
        [26.53151, 18.80121],
        [30.15454, 18.06146],
        [33.77756, 17.52064],
        [37.40059, 17.11480],
        [41.02361, 16.77292],
        [44.64663, 16.48787],
        [48.26966, 16.22413],
        [51.89268, 15.95329],
        [55.51571, 15.78902],
        [59.13873, 15.63896],
        [62.76176, 15.36101],
        [66.38478, 14.89124],
        [70.00780, 14.44278],
        [73.66377, 14.15756],
        [77.63027, 13.95785],
        [80.94275, 13.87604],
        [84.49990, 13.62229],
        [86.80546, 13.63595]
    ]

    b04 = [
        [-86.01027, 16.65277],
        [-82.25803, 17.08352],
        [-79.19493, 17.31579],
        [-74.97904, 17.71826],
        [-71.29015, 18.07132],
        [-67.66712, 18.73831],
        [-64.04410, 19.63976],
        [-60.42107, 20.50568],
        [-56.79805, 21.23661],
        [-53.17502, 21.94623],
        [-49.55200, 22.60611],
        [-45.92898, 23.03864],
        [-42.30595, 23.30776],
        [-38.68293, 23.76161],
        [-35.05990, 24.47123],
        [-31.43688, 25.37978],
        [-27.97854, 26.49479],
        [-23.20273, 27.92891],
        [-19.57971, 29.29880],
        [-15.95668, 30.34234],
        [-12.33366, 31.02354],
        [-8.71063, 31.51291],
        [-5.08761, 31.81045],
        [-1.46459, 31.97300],
        [2.15844, 32.01477],
        [5.78146, 31.87181],
        [9.40449, 31.56545],
        [13.02751, 31.02463],
        [16.65054, 30.30618],
        [20.10888, 29.33177],
        [23.48488, 27.70503],
        [27.19024, 26.26107],
        [30.81327, 24.87478],
        [34.43629, 23.87925],
        [38.05932, 23.08976],
        [41.68234, 22.54894],
        [45.30537, 22.11469],
        [48.92839, 21.68044],
        [52.55141, 21.24619],
        [56.17444, 20.83325],
        [59.79746, 20.39189],
        [63.42049, 19.90791],
        [67.04351, 19.50208],
        [70.66654, 19.13888],
        [74.35543, 18.81130],
        [78.42493, 18.41836],
        [82.52371, 18.31191],
        [85.65268, 18.09092],
    ]

    b06 = [
        [-86.97620, 20.59368],
        [-83.80605, 21.10092],
        [-80.18302, 21.77502],
        [-76.72468, 22.51262],
        [-72.60761, 23.39063],
        [-68.98459, 24.33008],
        [-65.36156, 25.56545],
        [-61.73854, 26.90028],
        [-58.11551, 28.01487],
        [-54.49249, 29.07263],
        [-50.86946, 29.95276],
        [-47.24644, 30.52738],
        [-43.62341, 30.86045],
        [-40.00039, 31.17930],
        [-36.37737, 31.58342],
        [-32.75434, 32.13673],
        [-29.29600, 32.74834],
        [-26.00234, 33.43248],
        [-21.88527, 34.31623],
        [-18.26224, 34.96809],
        [-14.63922, 35.61377],
        [-11.01620, 35.96815],
        [-7.39317, 36.22306],
        [-3.77015, 36.42824],
        [-0.14712, 36.59789],
        [3.47590, 36.68940],
        [7.09893, 36.70985],
        [10.72195, 36.58821],
        [14.34498, 36.35289],
        [17.96800, 35.96838],
        [21.59102, 35.33519],
        [25.21405, 34.22883],
        [28.71356, 32.92344],
        [31.06029, 32.13322],
        [34.43629, 30.89877],
        [38.05932, 29.93166],
        [41.68234, 29.03560],
        [45.30537, 28.53030],
        [48.92839, 28.10315],
        [52.55141, 27.64048],
        [56.17444, 27.14939],
        [59.79746, 26.49489],
        [63.42049, 25.64146],
        [67.04351, 24.96565],
        [70.66654, 24.38930],
        [74.28956, 23.96216],
        [78.02237, 23.58945],
        [81.74146, 23.24174],
        [85.25744, 23.04119],
    ]

    b08 = [
        [-86.01027, 27.59411],
        [-82.15922, 28.34626],
        [-77.87746, 29.13530],
        [-74.25444, 29.83753],
        [-70.63141, 30.46899],
        [-67.00839, 31.32070],
        [-63.38537, 32.32162],
        [-59.76234, 33.26569],
        [-56.13932, 34.11740],
        [-52.51629, 34.92649],
        [-48.89327, 35.59348],
        [-45.27024, 36.12548],
        [-41.64722, 36.59353],
        [-38.02420, 37.09711],
        [-34.40117, 37.65753],
        [-30.77815, 38.23405],
        [-26.78459, 38.93074],
        [-23.53210, 39.42404],
        [-19.90907, 40.12655],
        [-16.28605, 40.60171],
        [-12.66302, 40.94188],
        [-9.04000, 41.07601],
        [-5.41698, 41.19593],
        [-1.79395, 41.31585],
        [1.82907, 41.46419],
        [5.45210, 41.65516],
        [9.07512, 41.82482],
        [12.69815, 41.93763],
        [16.32117, 41.94388],
        [19.94420, 41.87197],
        [23.56722, 41.25726],
        [27.15731, 40.50577],
        [30.81327, 39.63849],
        [34.43629, 38.12432],
        [38.05932, 36.73092],
        [41.68234, 35.48673],
        [45.30537, 34.76118],
        [48.92839, 34.21325],
        [52.55141, 33.75059],
        [56.17444, 33.27371],
        [59.79746, 32.76841],
        [63.42049, 32.12812],
        [67.04351, 31.40257],
        [70.66654, 30.69123],
        [74.28956, 29.97607],
        [78.09557, 29.25466],
        [81.53561, 28.64248],
        [85.15863, 27.89562],
    ]

    b10 = [
        [-86.01027, 36.28708],
        [-82.15922, 37.00101],
        [-78.53620, 37.45486],
        [-74.83998, 37.86685],
        [-71.29015, 38.20624],
        [-67.66712, 38.79508],
        [-64.04410, 39.52601],
        [-60.42107, 40.45588],
        [-56.79805, 41.48521],
        [-53.17502, 42.57848],
        [-49.55200, 43.85648],
        [-45.92898, 45.14159],
        [-42.30595, 46.58300],
        [-38.68293, 47.86810],
        [-35.05990, 49.01821],
        [-31.43688, 49.83440],
        [-27.81385, 50.37137],
        [-23.53210, 50.56432],
        [-19.90907, 50.69135],
        [-16.28605, 50.64075],
        [-12.66302, 50.47648],
        [-9.04000, 50.24827],
        [-5.41698, 50.11952],
        [-1.79395, 50.07604],
        [1.82907, 50.08939],
        [5.45210, 50.32298],
        [9.07512, 50.62763],
        [12.69815, 50.94648],
        [16.32117, 51.40033],
        [19.94420, 51.78313],
        [23.56722, 51.93857],
        [27.00726, 51.89276],
        [30.99625, 51.50974],
        [34.43629, 50.77081],
        [38.05932, 49.72555],
        [41.68234, 48.24690],
        [44.97600, 46.57413],
        [48.10498, 44.91395],
        [51.56332, 43.34937],
        [55.18634, 42.07675],
        [58.80937, 41.20911],
        [62.43239, 40.56882],
        [66.05541, 40.13457],
        [69.67844, 39.77137],
        [73.30146, 39.42237],
        [77.11270, 38.97307],
        [80.54751, 38.63913],
        [84.17054, 38.22619],
        [86.73959, 37.89450],
    ]

    tropopause = [
        [-85.69224459466632, 8.7659753129117],
        [-79.63251933664336, 8.86646902536853],
        [-73.5727940786204, 8.86646902536853],
        [-67.51306882059745, 8.86646902536853],
        [-61.453343562574496, 8.86646902536853],
        [-55.78456832119818, 9.35090025464757],
        [-49.92031807149854, 10.072393574850409],
        [-44.05606782179891, 10.824808037347651],
        [-38.77824259706924, 12.094636280904645],
        [-33.10946735569293, 13.226350088879961],
        [-27.636167122639932, 14.205519594869529],
        [-21.576441864616967, 14.52503806524507],
        [-15.516716606594017, 14.52503806524507],
        [-9.65246635689438, 14.52503806524507],
        [-3.5927410988714286, 14.52503806524507],
        [2.4669841591515365, 14.52503806524507],
        [8.526709417174487, 14.52503806524507],
        [14.586434675197452, 14.52503806524507],
        [20.646159933220403, 14.52503806524507],
        [26.705885191243354, 14.277668926889817],
        [31.20181038267974, 12.586282443385727],
        [36.479635607409406, 11.103098317997315],
        [41.36651081549245, 9.643362725515509],
        [47.62171108183871, 9.573790155353088],
        [53.48596133153835, 8.866469025368524],
        [59.54568658956133, 8.866469025368517],
        [65.60541184758428, 8.878064453728918],
        [71.66513710560723, 8.866469025368517],
        [77.52938735530687, 8.866469025368517],
        [83.87146540313019, 9.010767689409079]
    ]

    tropopause_df = pd.DataFrame(tropopause, columns=["latitude", "altitude"])
    b02_df = pd.DataFrame(b02, columns=["latitude", "altitude"])
    b04_df = pd.DataFrame(b04, columns=["latitude", "altitude"])
    b06_df = pd.DataFrame(b06, columns=["latitude", "altitude"])
    b08_df = pd.DataFrame(b08, columns=["latitude", "altitude"])
    b10_df = pd.DataFrame(b10, columns=["latitude", "altitude"])

    # Add a 'value' column to each one before concatenating
    tropopause_df["value"] = 0.0
    b02_df["value"] = 0.2
    b04_df["value"] = 0.4
    b06_df["value"] = 0.6
    b08_df["value"] = 0.8
    b10_df["value"] = 1.0
    tropopause_df["source"] = "tropopause"
    b02_df["source"] = "b02"
    b04_df["source"] = "b04"
    b06_df["source"] = "b06"
    b08_df["source"] = "b08"
    b10_df["source"] = "b10"

    # Concatenate them
    b_df = pd.concat([tropopause_df, b02_df, b04_df, b06_df, b08_df, b10_df], ignore_index=True)

    # Add cornerpoints to fill above value = 1.0 line
    added_cornerpoints = [
        {"latitude": -87.0, "altitude": 60, "value": 1.0},
        {"latitude": 87.0, "altitude": 60, "value": 1.0}
    ]
    b_df = pd.concat([b_df, pd.DataFrame(added_cornerpoints)], ignore_index=True)
    b_df['altitude'] = b_df['altitude']*1000 # convert to meters
    return b_df

def get_griddata(b_df, heights, latitudes, plot_data = False):
    # Extract columns
    x = b_df['latitude'].values
    y = b_df['altitude'].values/1000 # due to griddata
    z = b_df["value"].astype(float).values  # z: value

    # Create grid
    # xi = np.linspace(-85, 86, 171)
    # yi = np.linspace(0, 60, 601)
    xi = latitudes
    yi = heights/1000 #due to gridddata
    X, Y = np.meshgrid(xi, yi)

    # Interpolate values onto grid
    myhre_2b_griddata = griddata((x, y), z, (X, Y), method='linear')

    # Make a Plot if plot_data is true
    if plot_data:
        plt.figure(figsize=(10, 6))
        heatmap = plt.pcolormesh(X, Y, myhre_2b_griddata, shading='auto', cmap='viridis')
        plt.colorbar(heatmap, label='Value')

        plt.xlabel('Altitude')
        plt.ylabel('Latitude')
        plt.title('CH4 ppmv')
        plt.tight_layout()
        plt.scatter(b_df[b_df["source"] == "b10"]["latitude"], b_df[b_df["source"] == "b10"]["altitude"]/1000, color="red",
                    label="b10") # /1000 due to griddata
        plt.show()
    return myhre_2b_griddata




def get_total_mass(heights, latitudes, delta_h, delta_deg):
    volumes = get_volume_matrix(heights, latitudes, delta_h, delta_deg)
    myhre_2b_df = construct_myhre_2b_df()
    myhre_grid = get_griddata(myhre_2b_df, heights, latitudes,)# plot_data=True)
    print("nansum thise",np.nansum(myhre_grid))
    print("nansum other", 56145.40940439073)

    number_density = Atmosphere(heights).number_density
    parts_mat = volumes * number_density[:, np.newaxis]

    SWV_mol_mat = parts_mat * myhre_grid * 10 ** -6 / N_avogrado
    SWV_mass_mat = SWV_mol_mat * M_h2o
    total_SWV_mass = np.nansum(SWV_mass_mat)
    print("Total SWV mass is:", total_SWV_mass * 10 ** -9, "Tg")
    return total_SWV_mass

def get_SWV_RF(total_SWV_mass):
    # based on the formula of pletzer
    total_SWV_mass_Tg = total_SWV_mass * 10 ** -9
    # only valid when the mass is in a decent range:
    # 0-160 Tg as the plot reaches till 160Tg
    if total_SWV_mass_Tg < 0.0 or total_SWV_mass_Tg > 160:
        raise ValueError("Total SWV mass out of range of Pletzer plot")
    a = -0.00088
    b = 0.47373
    c = -0.74676
    SWV_RF = a*total_SWV_mass_Tg**2 + b*total_SWV_mass_Tg + c # in mW for mass in Tg
    return SWV_RF * 10**-3 # W/m**2


delta_h = 100.        # height increment in meters
delta_deg = 1.        # latitude increment
heights = np.arange(0, 60000 + delta_h, delta_h)  # 0 to 60 km
latitudes = np.arange(-85, 86, delta_deg)  # -85° to 85°
x = get_total_mass(heights, latitudes, delta_h, delta_deg)
print(x)

print(get_SWV_RF(x))
# difference with the jupyter way of computing
print(get_SWV_RF(130.30558591104*10**9))
# print(get_SWV_RF(-170.30558591104*10**9))



### OLD CODE FROM BEFORE SUMMER
# def calc_stratospheric_ch4_loss(conc_ch4, lifetime_ch4 , mass_atm):
#     STRATOSPHERIC_LOSS_FACTOR = 0.08
#     mass_ch4 = mass_atm * conc_ch4 * (molar_mass_ch4 / molar_mass_air)
#     loss_ch4 = mass_ch4 / lifetime_ch4
#     stratospheric_loss_ch4 = loss_ch4 * STRATOSPHERIC_LOSS_FACTOR
#     return stratospheric_loss_ch4
#
#
# def calc_swv_mass(mass_stratospheric_loss_ch4):
#     parts_stratospheric_loss_ch4 =  mass_stratospheric_loss_ch4/molar_mass_ch4
#     parts_swv = parts_stratospheric_loss_ch4*2
#     mass_swv = parts_swv * molar_mass_h2o
#     return mass_swv
#
#
#
# # since 1750 to 2000 the ch4 concentrations go from 700 to 1750 ppbv from myhre and the background files
#
# def trial_run():
#     old_trial_conc_ch4 = 750 * 10**-9 # 750 ppbv
#     trial_lifetime_ch4 = 8 # yr
#     trial_mass_atm = 5.15 *10**18 #kg
#     old_trial_strat_ch4_loss = calc_stratospheric_ch4_loss(old_trial_conc_ch4, trial_lifetime_ch4, trial_mass_atm)
#     old_trial_mass_swv = calc_swv_mass(old_trial_strat_ch4_loss)
#
#     new_trial_conc_ch4 = 1750 * 10**-9 # 1750 ppbv
#     new_trial_strat_ch4_loss = calc_stratospheric_ch4_loss(new_trial_conc_ch4, trial_lifetime_ch4, trial_mass_atm)
#     new_trial_mass_swv = calc_swv_mass(new_trial_strat_ch4_loss)
#     return old_trial_mass_swv, new_trial_mass_swv
#
#
#
# old, new = trial_run()
# delta_swv_mass = new - old
# print(delta_swv_mass,"kg or ", delta_swv_mass*10**-9," Tg")
# # 64 Tg -> ~26mW/m2 (pletzer), 28 (grewe), myhre said 83...












