import numpy as np

def _valuta_funzione(f, x):
    """
    Helper interno: assicura che la valutazione f(x) restituisca sempre 
    un array NumPy della stessa dimensione di x, anche per funzioni costanti.
    """
    y = f(x)
    if np.isscalar(y) or (isinstance(y, np.ndarray) and y.ndim == 0):
        return np.full_like(x, y, dtype=float)
    return np.asarray(y, dtype=float)

def trapcomp(f, a, b, N):
    """
    Calcola l'integrale definito usando il metodo dei trapezi composito.

    Parametri:
    ----------
    f : callable
        Funzione matematica da integrare.
    a, b : float
        Estremi dell'intervallo di integrazione.
    N : int
        Numero di sottointervalli (N >= 1).

    Ritorna: float
    """
    if not callable(f):
        raise TypeError("Il parametro 'f' deve essere una funzione (callable).")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("Il parametro 'N' deve essere un intero positivo (N >= 1).")
    
    a, b = float(a), float(b)
    if a == b:
        return 0.0

    h = (b - a) / N
    nodi = np.linspace(a, b, N + 1)
    
    y = _valuta_funzione(f, nodi)

    I = (h / 2.0) * np.sum(y[:-1] + y[1:])

    return I


def pmcomp(f, a, b, N):
    """
    Calcola l'integrale definito usando il metodo del punto medio composito.

    Parametri:
    ----------
    f : callable
        Funzione matematica da integrare.
    a, b : float
        Estremi dell'intervallo di integrazione.
    N : int
        Numero di sottointervalli (N >= 1).

    Ritorna: float
    """
    if not callable(f):
        raise TypeError("Il parametro 'f' deve essere una funzione (callable).")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("Il parametro 'N' deve essere un intero positivo (N >= 1).")
    
    a, b = float(a), float(b)
    if a == b:
        return 0.0

    h = (b - a) / N
    punti_medi = np.linspace(a + h/2.0, b - h/2.0, N)
    
    y_m = _valuta_funzione(f, punti_medi)
    I = h * np.sum(y_m)

    return I

def simpcomp(f, a, b, N):
    """
    Calcola l'integrale definito usando il metodo di Simpson composito.

    Parametri:
    ----------
    f : callable
        Funzione matematica da integrare.
    a, b : float
        Estremi dell'intervallo di integrazione.
    N : int
        Numero di sottointervalli (N >= 1).

    Ritorna: float
    """
    if not callable(f):
        raise TypeError("Il parametro 'f' deve essere una funzione (callable).")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("Il parametro 'N' deve essere un intero positivo (N >= 1).")
    
    a, b = float(a), float(b)
    if a == b:
        return 0.0

    h = (b - a) / N
    punti_medi = np.linspace(a + h/2.0, b - h/2.0, N)
    nodi = np.linspace(a, b, N + 1)
    
    y = _valuta_funzione(f, nodi)
    y_m = _valuta_funzione(f, punti_medi)
    
    I = (h / 6.0) * np.sum(4.0 * y_m + y[:-1] + y[1:])

    return I