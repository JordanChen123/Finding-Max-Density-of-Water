import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

x = [-10, 0, 10, 20, 30]
y = [0.99815, 0.99987, 0.99973, 0.99823, 0.99567]
     
f = CubicSpline(x, y, bc_type='natural')
x_new = np.linspace(-10, 30, 1000)  # More points for smoother interpolation
y_new = f(x_new)

plt.figure(figsize=(10, 8))
plt.plot(x_new, y_new, 'b')
plt.plot(x, y, 'ro')
plt.title('Cubic Spline Interpolation')
plt.xlabel('Temperature (Degrees Celcius)')
plt.ylabel('Density (g/cm^3)')
plt.show()

def quadratic_spline_roots(spl): 
    roots = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a+b)/2), spl(b)
        t = np.roots([u+w-2*v, w-u, 2*v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t*(b-a)/2 + (b+a)/2)
    return np.array(roots)

from scipy.interpolate import InterpolatedUnivariateSpline

f = InterpolatedUnivariateSpline(x, y, k=3)
cr_pts = quadratic_spline_roots(f.derivative())
cr_pts = np.append(cr_pts, (x[0], x[-1]))  
cr_vals = f(cr_pts)
min_index = np.argmin(cr_vals)
max_index = np.argmax(cr_vals)
print("Maximum value {} at {}\nMinimum value {} at {}".format(cr_vals[max_index], cr_pts[max_index], cr_vals[min_index], cr_pts[min_index]))
