import numpy as np


def naca_4digit_airfoil(naca_code, c=1.0, num_points=100,cluster=True):
    """
    Generates the coordinates for a NACA 4-digit airfoil.

    Parameters:
    naca_code (str or int): The 4-digit NACA code (e.g., '2412' or 2412).
    c (float): The chord length (default is 1.0).
    num_points (int): The number of points for each surface (upper and lower).
    cluster (bool): Enables clustering around LE and TE

    Returns:
    tuple: (xu, yu, xl, yl, x, yc) coordinates for the upper and lower surfaces.
    """
    if isinstance(naca_code, int):
        naca_code = str(naca_code)
    
    m = int(naca_code[0]) / 100.0   # Maximum camber (as a fraction of chord)
    p = int(naca_code[1]) / 10.0    # Location of maximum camber (as a fraction of chord)
    t = int(naca_code[2:]) / 100.0  # Maximum thickness (as a fraction of chord)

    
    if cluster:
        # For better definition around the leading edge
        x = (c / 2.0) * (1.0 - np.cos(np.linspace(0, np.pi, num_points)))
    else:
        # linear spacing
        x = np.linspace(0, c, num_points)

    # Half thickness equation
    yt = (t * c *5) * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)
    
    # Camber line and gradient
    yc = np.zeros(num_points)
    dyc_dx = np.zeros(num_points)
    
    for i in range(num_points):
        if x[i] < p * c:
            yc[i] = (m / p**2) * (2 * p * (x[i] / c) - (x[i] / c)**2)
            dyc_dx[i] = (2 * m / p**2) * (p - (x[i] / c))
        else:
            yc[i] = (m / (1 - p)**2) * (1 - 2 * p + 2 * p * (x[i] / c) - (x[i] / c)**2)
            dyc_dx[i] = (2 * m / (1 - p)**2) * (p - (x[i] / c))

    # Angle of the camber line gradient (theta)
    theta = np.arctan(dyc_dx)

    # Upper and lower surface coordinates
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    return xu, yu, xl, yl, x, yc



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example Usage:
    naca_code = '2412'
    xu, yu, xl, yl, x, yc = naca_4digit_airfoil(naca_code)

    # Plotting the airfoil
    plt.figure(figsize=(10, 5))
    plt.plot(xu, yu, 'b-', label='Upper Surface')
    plt.plot(xl, yl, 'r-', label='Lower Surface')
    plt.plot(x, yc, 'k--', label='Camber line')
    plt.title(f'NACA {naca_code} Airfoil')
    plt.xlabel('X/c (Chord position)')
    plt.ylabel('Y/c (Thickness position)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()
