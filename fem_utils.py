import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# GRID CLASS
# ============================================================

class Grid:
    def __init__(self, a, b, N):
        """
        Initialize a 1D uniform grid on the interval [a, b] with N elements.

        Args:
            a (float): Left endpoint.
            b (float): Right endpoint.
            N (int): Number of elements.
        """
        self.a = a
        self.b = b
        self.N = N

    def compute_geometry(self):
        """
        Compute mesh spacing and nodal coordinates for the current grid.

        Returns:
            None.
        """
        self.h = (self.b - self.a) / self.N
        # Coordinates of all nodal points, including both endpoints.
        self.nodes = np.linspace(self.a, self.b, self.N + 1)

    def plot(self):
        """
        Plot the 1D mesh nodes on a horizontal line.

        Returns:
            None.
        """
        plt.figure(figsize=(8, 1))
        # Display nodes as markers with y=0 to visualize the 1D mesh.
        plt.plot(self.nodes, np.zeros(self.N + 1), marker="o")

        plt.title("Mesh")
        plt.axis("off")
        plt.show()


def fun2dof(grid, f):
    """
    evaluate a function at grid nodes.

    Args:
        grid (Grid): Finite element grid.
        f (callable): Continuous function.

    Returns:
        np.ndarray: Nodal values.
    """
    return f(grid.nodes)

def diffusion(grid):
    """
    Assemble the 1D P1 finite-element diffusion matrix on a uniform grid.

    Args:
        grid (Grid): Finite element grid.

    Returns:
        np.ndarray: Diffusion matrix of shape (N+1, N+1).
    """
    # Main diagonal
    main = 2.0 * np.ones(grid.N + 1)
    # Boundary basis functions have support on one element only.
    main[0] = 1.0
    main[-1] = 1.0
    # off  diagonal 
    off = -1.0 * np.ones(grid.N + 1 - 1)
    # Build the full matrix from its diagonals.
    A = np.diag(main) + np.diag(off, k=-1) + np.diag(off, k=1)
    # Scale by 1/h to account for physical element size.
    return A / grid.h

def transport(grid):
    """
    Assemble the 1D P1 finite-element transport matrix on a uniform grid.

    Args:
        grid (Grid): Finite element grid.

    Returns:
        np.ndarray: Transport matrix of shape (N+1, N+1).
    """
    # Local element matrix for int_K phi_i' * phi_j dx on a P1 interval.
    main = np.zeros(grid.N + 1)
    lower = -0.5 * np.ones(grid.N + 1 - 1)
    upper = 0.5 * np.ones(grid.N + 1 - 1)
    # Assemble the global tridiagonal transport stencil.
    T = np.diag(main) + np.diag(lower, k=-1) + np.diag(upper, k=1)
    return T
    

def mass(grid):
    """
    Assemble the 1D P1 finite-element mass matrix on a uniform grid.

    Args:
        grid (Grid): Finite element grid.

    Returns:
        np.ndarray: Mass matrix of shape (N+1, N+1).
    """
    # Consistent mass matrix stencil for linear elements.
    main = 2.0 / 3.0 * np.ones(grid.N + 1)
    # Boundary basis functions live on a single element, so their diagonal
    # contribution is half of the interior one.
    main[0] = 1.0 / 3.0
    main[-1] = 1.0 / 3.0
    off = 1.0 / 6.0 * np.ones(grid.N + 1 - 1)
    # Build the full matrix from its diagonals.
    M = np.diag(main) + np.diag(off, k=-1) + np.diag(off, k=1)
    # Scale by h for the physical element size.
    return M * grid.h

def create_restriction(keep_dof):
    """
    Create the restriction matrix that keeps only selected degrees of freedom.

    Args:
        keep_dof: Boolean array indicating which degrees of freedom (dofs)
            to keep. True for the dofs of the system, False for the overwritten values.

    Returns:
        Restriction mapping matrix as a NumPy array.
    """
    # Convert boolean mask to diagonal selector.
    R = np.diag(keep_dof.astype(int))
    # Keep only the rows associated with active degrees of freedom.
    return R[keep_dof, :]


def simpcomp(f, a, b, N):
    """
    Composite Simpson quadrature.

    Args:
        f (callable): Function to integrate.
        a (float): Left boundary.
        b (float): Right boundary.
        N (int): Number of subintervals.

    Returns:
        float: Approximated integral.
    """
    # Calcolo della larghezza di ogni sottointervallo
    h = (b - a) / N

    # Griglia spaziale nei nodi
    x = np.linspace(a, b, N + 1)

    # Nodi sinistri e destri di ogni sottointervallo
    xL, xR = x[:-1], x[1:]

    # Punti medi dei sottointervalli
    xM = 0.5 * (xL + xR)

    # Calcolo dell'integrale approssimato usando la formula di Simpson
    I = (h / 6.0) * (f(xL) + 4 * f(xM) + f(xR)).sum()

    return I

def error_l2(grid, u_ex, u_h, rel=False):
    """
    Compute discrete L2 error using Simpson quadrature.

    Args:
        grid (Grid): Finite element grid.
        u_ex (callable): Exact solution.
        u_h (callable): Numerical solution.
        rel (bool): Compute relative error if True.

    Returns:
        float: L2 error.
    """
    
    # creo la funzione integranda da integrare
    integrand = lambda x: (u_ex(x) - u_h(x)) ** 2
    integral = simpcomp(integrand, grid.a, grid.b, grid.N)
    if rel:
        # Normalizzo per ottenere l'errore relativo
        norm_ex = simpcomp(lambda x: u_ex(x) ** 2, grid.a, grid.b, grid.N)
        integral /= norm_ex
    
    return np.sqrt(integral)
