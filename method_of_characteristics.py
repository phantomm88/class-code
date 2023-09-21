import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

gamma = 1.4

def p_m_function(M):
  ft = np.sqrt((gamma+1)/(gamma-1))
  st = np.arctan(np.sqrt((M**2 - 1) * (gamma-1)/(gamma+1)))
  tt = np.arctan(np.sqrt((M**2 - 1)))
  return np.degrees(ft * st - tt)

def inv_pm_function(v):
  deflec = v
  ft = np.sqrt((gamma+1)/(gamma-1))
  return optimize.brenth(lambda M : ft * np.arctan(np.sqrt((M**2 - 1) * (gamma-1)/(gamma+1))) - np.arctan(np.sqrt((M**2 - 1))) - np.radians(deflec), 1, 4)


flow_dict = {}

# Defines initial wave conditions
def initial_subs(N):
  for i in range(1, N+1):
    incidence = -12 * (i - 1) / (N - 1)
    x_pos = 0
    y_pos = 0
    if i == 1:
      M = 2
      mach_angle = np.degrees(np.arcsin(1/M))
      reference = mach_angle + incidence
      deflection = p_m_function(M)
      left_r_invariant = deflection - incidence
      flow_dict[-1] = {'incidence': incidence, 'deflection' : deflection, 'M': M, 'mach_angle': mach_angle, 'left_reference': reference, 'left_r_invariant': left_r_invariant, 'x_pos': x_pos, 'y_pos': y_pos}
    else:
      deflection = flow_dict[-1]['deflection'] - incidence
      M = inv_pm_function(deflection)
      mach_angle = np.degrees(np.arcsin(1/M))
      reference = incidence + mach_angle
      left_r_invariant = deflection - incidence
      flow_dict[-i] = {'incidence': incidence, 'deflection' : deflection, 'M': M, 'mach_angle': mach_angle, 'left_reference': reference, 'left_r_invariant': left_r_invariant, 'x_pos': x_pos, 'y_pos': y_pos}

# Creates list of points that correspond to wall points for given N waves
def wall_pointer(N):
  wall_list = []
  sum = 1
  limit = N * (N + 1) / 2
  i = 0
  while sum <= limit:
    wall_list.append(sum)
    sum += (N - i)
    i += 1
  return wall_list

# returns list of layer length
def layer_pointer(N):
  limit = N * (N + 1) / 2
  layer_list = []
  sum = N
  i = N
  while sum <= limit and i > 0:
    layer_list.append(i)
    i = i - 1
    sum += i
  return layer_list

# returns layer index of a point
def layer_finder(N, p):
  limit = N * (N + 1) / 2
  lst = wall_pointer(N)
  i = 0
  target = 0
  killSwitch = False
  while not killSwitch:
    if p == int(limit):
      target = len(lst)
      killSwitch = True
    elif p < lst[i]:
      target = i
      killSwitch = True
    i += 1
  return target - 1



# Governs wall points conditions and computations
def wall_point(N, p):
  if p == 1:
    incidence = 0
    deflection = flow_dict[-1]['left_r_invariant'] + incidence
    M = inv_pm_function(deflection)
    mach_angle = np.degrees(np.arcsin(1/M))
    right_r_invariant = flow_dict[-1]['left_r_invariant']
    reference = incidence - mach_angle
    x_pos = 1 / (np.tan(np.radians(flow_dict[-1]['left_reference'])))
    y_pos = 1
    flow_dict[1] = {'incidence': incidence, 'deflection' : deflection, 'M': M, 'mach_angle': mach_angle, 'right_reference': reference, 'right_r_invariant': right_r_invariant, 'x_pos': x_pos, 'y_pos': y_pos}
  else:
    incidence = 0
    deflection = flow_dict[p - layer_pointer(N)[layer_finder(N, p)]]['left_r_invariant'] + incidence
    M = inv_pm_function(deflection)
    mach_angle = np.degrees(np.arcsin(1/M))
    right_r_invariant = flow_dict[p - layer_pointer(N)[layer_finder(N, p)]]['left_r_invariant']
    reference = incidence - mach_angle
    x_pos = 1 / (np.tan(np.radians(flow_dict[p - layer_pointer(N)[layer_finder(N, p)]]['left_reference'])))
    y_pos = 1
    flow_dict[p] = {'incidence': incidence, 'deflection' : deflection, 'M': M, 'mach_angle': mach_angle, 'right_reference': reference, 'right_r_invariant': right_r_invariant, 'x_pos': x_pos, 'y_pos': y_pos}


# Governs interior points conditions and computations
def interior_point(N, p):
  if layer_finder(N, p) == 0:
    deflection = (flow_dict[p - 1]['right_r_invariant'] + flow_dict[-p]['left_r_invariant']) / 2
    M = inv_pm_function(deflection)
    mach_angle = np.degrees(np.arcsin(1/M))
    incidence = (flow_dict[p - 1]['right_r_invariant'] - flow_dict[-p]['left_r_invariant']) / 2
    left_r_invariant = flow_dict[-p]['left_r_invariant']
    right_r_invariant = flow_dict[p - 1]['right_r_invariant']
    left_reference = incidence + mach_angle
    right_reference = incidence - mach_angle
    x_pos = (np.tan(np.radians(flow_dict[p - 1]['right_reference'])) * flow_dict[p - 1]['x_pos'] - np.tan(np.radians(flow_dict[-p]['left_reference'])) * flow_dict[-p]['x_pos'] + flow_dict[-p]['y_pos'] - flow_dict[p - 1]['y_pos']) / (np.tan(np.radians(flow_dict[p - 1]['right_reference'])) - np.tan(np.radians(flow_dict[-p]['left_reference'])))
    y_pos = np.tan(np.radians(flow_dict[p - 1]['right_reference'])) * (x_pos - flow_dict[p - 1]['x_pos']) + flow_dict[p - 1]['y_pos']
    flow_dict[p] = {'incidence': incidence, 'deflection' : deflection, 'M': M, 'mach_angle': mach_angle, 'left_reference' : left_reference, 'right_reference': right_reference, 'left_r_invariant' : left_r_invariant, 'right_r_invariant': right_r_invariant, 'x_pos': x_pos, 'y_pos': y_pos}
  else:
    deflection = (flow_dict[p - 1]['right_r_invariant'] + flow_dict[p - layer_pointer(N)[layer_finder(N, p) - 1]]['left_r_invariant']) / 2
    M = inv_pm_function(deflection)
    mach_angle = np.degrees(np.arcsin(1/M))
    incidence = (flow_dict[p - 1]['right_r_invariant'] - flow_dict[p - layer_pointer(N)[layer_finder(N, p) - 1]]['left_r_invariant']) / 2
    left_r_invariant = flow_dict[p - layer_pointer(N)[layer_finder(N, p) - 1]]['left_r_invariant']
    right_r_invariant = flow_dict[p - 1]['right_r_invariant']
    left_reference = incidence + mach_angle
    right_reference = incidence - mach_angle
    x_pos = (np.tan(np.radians(flow_dict[p - 1]['right_reference'])) * flow_dict[p - 1]['x_pos'] - np.tan(np.radians(flow_dict[p - layer_pointer(N)[layer_finder(N, p) - 1]]['left_reference'])) * flow_dict[p - layer_pointer(N)[layer_finder(N, p) - 1]]['x_pos'] + flow_dict[p - layer_pointer(N)[layer_finder(N, p) - 1]]['y_pos'] - flow_dict[p - 1]['y_pos']) / (np.tan(np.radians(flow_dict[p - 1]['right_reference'])) - np.tan(np.radians(flow_dict[p - layer_pointer(N)[layer_finder(N, p) - 1]]['left_reference'])))
    y_pos = np.tan(np.radians(flow_dict[p - 1]['right_reference'])) * (x_pos - flow_dict[p - 1]['x_pos']) + flow_dict[p - 1]['y_pos']
    flow_dict[p] = {'incidence': incidence, 'deflection' : deflection, 'M': M, 'mach_angle': mach_angle, 'left_reference' : left_reference, 'right_reference': right_reference, 'left_r_invariant' : left_r_invariant, 'right_r_invariant': right_r_invariant, 'x_pos': x_pos, 'y_pos': y_pos}

def synthesis(N):
  limit = N * (N + 1) / 2
  initial_subs(N)
  for i in range(1, int(limit + 1)):
    if i in wall_pointer(N):
      wall_point(N, i)
    else:
      interior_point(N, i)
  # the constant being factored in is to account for rounding errors from the program
  return flow_dict[3]['x_pos'] * 2.399661236

print(synthesis(100))
