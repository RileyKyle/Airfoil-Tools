import numpy as np

def naca_5digit_airfoil(naca_code, c=1.0, num_points=1000):
    """
    Generates the coordinates for a NACA 5-digit airfoil.

    Parameters:
    naca_code (str or int): The 5-digit NACA code (e.g., '23012' or 23012).
    c (float): The chord length (default is 1.0).
    num_points (int): The number of points for each surface (upper and lower).

    Returns:
    tuple: (xu, yu, xl, yl, x, yc) coordinates for the upper and lower surfaces.
    """
    if isinstance(naca_code, int):
        naca_code = str(naca_code)

    L = int(naca_code[0])           # Design Lift Coeff (Cl = 0.15 * L)
    P = int(naca_code[1])           # Max camber position index
    Q = int(naca_code[2])           # Standard (0) or Reflex (1)
    t = int(naca_code[3:]) / 100.0  # Max thickness
    
    # Mapping for standard non-reflexed series (Q=0)
    p_map = {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25}
    m_map = {1: 0.0580, 2: 0.1260, 3: 0.2025, 4: 0.2900, 5: 0.3910}
    k1_map = {1: 361.4, 2: 51.64, 3: 15.957, 4: 6.643, 5: 3.230}
    
    p = p_map[P]
    m = m_map[P]

    # k1 must be scaled by (Design Cl / 0.3)
    k1 = k1_map[P] * ( (0.15 * L) / 0.3 )

    # linear spacing
    x = np.linspace(0, c, num_points)

    # For better definition around the leading edge
    # x = (c / 2.0) * (1.0 - np.cos(np.linspace(0, np.pi, num_points)))

    # Half thickness equation
    yt = (t * c *5) * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)

    # Camber line and gradient
    yc = np.zeros(num_points)
    dyc_dx = np.zeros(num_points)
    
    for i in range(len(x)):
        if x[i] <= m:
            yc[i] = (k1/6.0) * ((x[i]/c)**3 - 3*m*(x[i]/c)**2 + (m**2)*(3-m)*x[i])
            dyc_dx[i] = (k1/6.0) * (3*(x[i]/c)**2 - 6*m*(x[i]/c) + (m**2)*(3-m))
        else:
            yc[i] = (k1 * (m**3) / 6.0) * (1.0 - (x[i]/c))
            dyc_dx[i] = -(k1 * (m**3) / 6.0)

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
    naca_code = '23012'
    xu, yu, xl, yl, x, yc = naca_5digit_airfoil(naca_code)

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
