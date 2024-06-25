import sympy as sm

# Define the symbolic variables
vx, vy, wz, delta, a, psi = sm.symbols('vx vy wz delta a psi')
X, Y = sm.symbols('X Y')
lf, lr = sm.symbols('lf lr')
m, Iz = sm.symbols('m Iz')
Df, Dr = sm.symbols('Df Dr')
Cf, Cr = sm.symbols('Cf Cr')
Bf, Br = sm.symbols('Bf Br')
s = sm.symbols('s')
ey = sm.symbols('ey')
cur = sm.symbols('kappa')
epsi = sm.symbols('epsi')
deltaT = sm.symbols('DeltaT')

# Compute tire split angle
alpha_f = delta - sm.atan2( vy + lf * wz, vx )
alpha_r = - sm.atan2( vy - lr * wz , vx)

# Compute lateral force at front and rear tire
Fyf = Df * sm.sin( Cf * sm.atan(Bf * alpha_f ) )
Fyr = Dr * sm.sin( Cr * sm.atan(Br * alpha_r ) )

f = sm.Matrix([
    vx   + deltaT *(a - 1 / m * Fyf * sm.sin(delta) + wz * vy),
    vy   + deltaT *(1 / m * (Fyf * sm.cos(delta) + Fyr) - wz * vx),
    wz   + deltaT *(1 / Iz * (lf * Fyf * sm.cos(delta) - lr * Fyr)),
    epsi + deltaT *(wz - (vx * sm.cos(epsi) - vy * sm.sin(epsi)) / (1 - cur * ey) * cur),
    s    + deltaT *((vx * sm.cos(epsi) - vy * sm.sin(epsi)) / (1 - cur * ey)),
    ey   + deltaT *(vx * sm.sin(epsi) + vy * sm.cos(epsi))
])

# Compute the jacobian of the system (A Matrix)
jac = f.jacobian([vx, vy, wz, epsi, s, ey])

# Print the jacobian element by element
for i in range(6):
    for j in range(6):
        print(f'jac[{i},{j}] = {jac[i,j]}')

# use sympy pprint to print the jacobian into a text file
sm.init_printing()

with open('jacobian.tex', 'w') as file:
    sm.print_latex(jac, ofile=file)

vxfunc = sm.lambdify((vx, vy, wz, epsi, delta, a, psi, X, Y, lf, lr, m, Iz, Df, Dr, Cf, Cr, Bf, Br, s, ey, cur, deltaT), jac[0,:], modules='numpy', cse=True)
vyfunc = sm.lambdify((vx, vy, wz, epsi, delta, a, psi, X, Y, lf, lr, m, Iz, Df, Dr, Cf, Cr, Bf, Br, s, ey, cur, deltaT), jac[1,:], modules='numpy', cse=True)
wzfunc = sm.lambdify((vx, vy, wz, epsi, delta, a, psi, X, Y, lf, lr, m, Iz, Df, Dr, Cf, Cr, Bf, Br, s, ey, cur, deltaT), jac[2,:], modules='numpy', cse=True)

# Compute the control jacobian of the system (B Matrix)
jac = f.jacobian([delta, a])

# Print the jacobian element by element
for i in range(6):
    for j in range(2):
        print(f'jac[{i},{j}] = {jac[i,j]}')

# use sympy pprint to print the jacobian into a text file
with open('control_jacobian.tex', 'w') as f:
    sm.print_latex(jac)

control_vxfunc = sm.lambdify((vx, vy, wz, epsi, delta, a, psi, X, Y, lf, lr, m, Iz, Df, Dr, Cf, Cr, Bf, Br, s, ey, cur, deltaT), jac[0,:], modules='numpy', cse=True)
# control_vyfunc = sm.lambdify((vx, vy, wz, epsi, delta, a, psi, X, Y, lf, lr, m, Iz, Df, Dr, Cf, Cr, Bf, Br, s, ey, cur, deltaT), jac[1,:], modules='numpy', cse=True)
# control_vyfunc = sm.lambdify((vx, vy, wz, epsi, delta, a, psi, X, Y, lf, lr, m, Iz, Df, Dr, Cf, Cr, Bf, Br, s, ey, cur, deltaT), jac[2,:], modules='numpy', cse=True)
myfunc = sm.lambdify((vx, vy, wz, epsi, delta, a, psi, X, Y, lf, lr, m, Iz, Df, Dr, Cf, Cr, Bf, Br, s, ey, cur, deltaT), jac[2,:], modules='numpy', cse=True)

