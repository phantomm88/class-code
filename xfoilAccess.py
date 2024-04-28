import os
import subprocess
import numpy as np
import GPyOpt
from GPyOpt.methods import BayesianOptimization

# assume a fixed angle of attack and fixed Re
alpha_i = 4.9
alpha_f = 5.1
alpha_step = 0.05
Re = 1e6
n_iter = 100

# x in this case will contain the four naca digits (max camber as %chord, distance of max camber from LE in 0.1chord, max thickness as %chord)
def objective_function(x):
    max_camb = x[:, 0]
    camb_dist = x[:, 1]
    max_thick = x[:, 2]
    pts = len(max_camb)

    ld_ratio_data = np.zeros(shape=(pts, 1))

    for i in range(0, pts):
        max_camb_str_i = str(int(max_camb[i]))
        camb_dist_str_i = str(int(camb_dist[i]))
        max_thick_str_i = str(int(max_thick[i]))

        name_i = 'NACA'+max_camb_str_i+camb_dist_str_i+max_thick_str_i

        #print(name_i)

        if os.path.exists("polar_file.txt"):
            os.remove("polar_file.txt")

        input_file = open("input_file.in", 'w')
        input_file.write("LOAD {0}.dat\n".format(name_i))
        input_file.write(name_i + '\n')
        input_file.write("PANE\n")
        input_file.write("OPER\n")
        input_file.write("Visc {0}\n".format(Re))
        input_file.write("PACC\n")
        input_file.write("polar_file.txt\n\n")
        input_file.write("ITER {0}\n".format(n_iter))
        input_file.write("ASeq {0} {1} {2}\n".format(alpha_i, alpha_f, alpha_step))
        input_file.write("\n\n")
        input_file.write("quit\n")
        input_file.close()

        subprocess.call("C:\\XFOIL6.99\\xfoil.exe < input_file.in", shell=True)

        # elements will be in order of
        # alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
        # alpha: angle of attack
        # CL: lift coefficient
        # CD: drag coefficient
        # CDp: pressure drag coefficient
        # Cm: pitching moment coefficient
        # Top_xtr: top position of forced transition from laminar to turbulent
        # Bot_xtr: bot position of forced transition from laminar to turbulent
        polar_data = np.loadtxt("polar_file.txt", skiprows=12)
        ld_ratios = np.array([el[1] / el[2] for el in polar_data])
        ld_ratio_avg = np.abs(np.mean(ld_ratios))

        if np.isnan(ld_ratio_avg):
            ld_ratio_data[i][0] = 0
        else:
            ld_ratio_data[i][0] = np.abs(ld_ratio_avg)
    
    return ld_ratio_data

#airfoil_stuff = np.array([[0, 0, 12], [1, 0, 18], [2, 6, 20], [1, 0, 50]])
#print(objective_function(airfoil_stuff))


bounds = [{'name': 'max_camb', 'type': 'continuous', 'domain': (0.0, 9.9)},
          {'name': 'camb_dist', 'type': 'continuous', 'domain': (0.0, 9.9)},
          {'name': 'max_thick', 'type': 'continuous', 'domain': (9.0, 50.00)}]

domain = bounds

max_iter = 250

optimizer = BayesianOptimization(f=objective_function,
                                 domain=domain,
                                 model_type='GP',
                                 acquisition_type='EI',
                                 acquisition_jitter=0.1,
                                 exact_feval=True,
                                 maximize=True)

optimizer.run_optimization(max_iter=max_iter, verbosity=False)

print()
print("Optimal design:", optimizer.x_opt)

N = int(optimizer.x_opt[0])
A = int(optimizer.x_opt[1])
CA = int(optimizer.x_opt[2])

print('This corresponds to a NACA '+str(N)+str(A)+str(CA)+' airfoil')
print("Optimal lift-to-drag ratio:", -optimizer.fx_opt)
print('This evaluation was done at an AOA of', (alpha_i+alpha_f)/2, 'degrees and a Reynolds number of', Re)

optimizer.plot_convergence()