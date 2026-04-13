"""
Modulo di utilità per la risoluzione di equazioni differenziali ordinarie (ODE).

Implementa i principali metodi numerici per la risoluzione di problemi ai valori iniziali:
- Metodo di Eulero in avanti (Forward Euler)
- Metodo di Eulero all'indietro (Backward Euler)
- Metodo di Crank-Nicolson
- Metodo del punto fisso
- Sostituzione in avanti e indietro per sistemi triangolari

Funzioni principali:
    - eulero_avanti: metodo esplicito di Eulero
    - eulero_indietro: metodo implicito di Eulero
    - crank_nicolson: metodo di Crank-Nicolson (semi-implicito)
    - puntofisso: metodo del punto fisso per iterazione funzionale
    - fwsub: sostituzione in avanti per matrici triangolari inferiori
    - bksub: sostituzione all'indietro per matrici triangolari superiori
"""

import numpy as np
from scipy.linalg import lu


def eulero_avanti(f, t0, tN, y0, h):
    """
    Metodo di Eulero in avanti (Forward Euler).

    Risolve il problema ai valori iniziali:
        y'(t) = f(t, y(t)),  t ∈ [t0, tN]
        y(t0) = y0

    utilizzando lo schema esplicito:
        y_{n+1} = y_n + h*f(t_n, y_n)

    Input:
        f (callable): Funzione che rappresenta il termine di destra dell'ODE: f(t, y)
        t0 (float): Tempo iniziale
        tN (float): Tempo finale
        y0 (float, list o numpy.ndarray): Condizione iniziale (scalare o vettore)
        h (float): Passo temporale

    Output:
        t_h (numpy.ndarray): Vettore degli istanti temporali (lunghezza N+1)
        u_h (numpy.ndarray): Soluzione discreta nei nodi temporali (matrice d × N+1)

    Avvertenza:
        Verificare che l'output di f e il dato y0 siano vettori della stessa lunghezza!

    Nota:
        Se la dimensione di y0 è 1, la soluzione è un array di lunghezza N+1.
        Il metodo è esplicito e facile da implementare ma ha requisiti stringenti
        sulla stabilità (CFL condition).
    """
    # Trasforma y0 in un vettore 1d
    y0 = np.atleast_1d(y0)

    # Determina il numero di passi temporali N e la dimensione di y0
    N = int((tN - t0) / h)
    d = len(y0)

    # Inizializza la matrice soluzione
    u_h = np.zeros((d, N+1))
    # Abbiamo N passi temporali, quindi N+1 nodi
    t_h = np.zeros(N+1)

    # Ciclo iterativo che calcola i passi di Eulero esplicito
    u_h[:, 0] = y0
    t_h[0] = t0

    for i in range(N):
        u_h[:, i+1] = u_h[:, i] + h*f(t_h[i], u_h[:, i])
        t_h[i+1] = t_h[i]+h

    if (d == 1):
        u_h = np.squeeze(u_h)

    return t_h, u_h


def eulero_indietro(f, t0, tN, y0, h, nmax_pf=300, toll_pf=1e-5):
    """
    Metodo di Eulero all'indietro (Backward Euler).

    Risolve il problema ai valori iniziali:
        y'(t) = f(t, y(t)),  t ∈ [t0, tN]
        y(t0) = y0

    utilizzando lo schema implicito:
        y_{n+1} = y_n + h*f(t_{n+1}, y_{n+1})

    Per problemi scalari, utilizza il metodo del punto fisso per risolvere l'equazione implicita.
    Per sistemi lineari, utilizza la fattorizzazione LU.

    Input:
        f (callable o numpy.ndarray): Termine di destra dell'ODE se scalare, 
                                     o matrice A se vettoriale (per sistemi lineari)
        t0 (float): Tempo iniziale
        tN (float): Tempo finale
        y0 (float, list o numpy.ndarray): Dato iniziale (scalare o vettore)
        h (float): Passo temporale
        nmax_pf (int): Numero massimo di iterazioni per il punto fisso (default: 300)
        toll_pf (float): Tolleranza per il criterio di arresto del punto fisso (default: 1e-5)

    Output:
        t_h (numpy.ndarray): Vettore degli istanti temporali (lunghezza N+1)
        u_h (numpy.ndarray): Soluzione discreta nei nodi temporali (matrice d × N+1)

    Nota:
        Il metodo è implicito e incondizionatamente stabile per problemi lineari.
        Adatto per problemi stiff.
    """

    # Assicura che y0 sia un vettore
    y0 = np.atleast_1d(y0)

    # Determina il numero di passi temporali
    N = int((tN - t0) / h)

    if (len(y0) == 1):
        # Caso scalare: utilizza il metodo del punto fisso

        # Inizializza la matrice soluzione
        t_h = np.zeros(N+1)     # Abbiamo N passi temporali, quindi N+1 nodi
        u_h = np.zeros(N+1)

        # Ciclo iterativo che calcola i passi
        u_h[0] = y0[0]
        t_h[0] = t0

        for i in range(N):
            # Definisce la funzione di punto fisso: φ(z) = u_n + h*f(t_{n+1}, z)
            def phi(z): return u_h[i] + h * f(t_h[i] + h, z)
            # Chiama il metodo del punto fisso
            u_pf = puntofisso(phi, u_h[i], nmax_pf, toll_pf)
            # Carica il vettore u
            u_h[i+1] = u_pf[-1]
            # Carica il vettore t
            t_h[i+1] = t_h[i]+h

        return t_h, u_h
    else:
        # Caso vettoriale (sistema lineare): y' = A*y

        # Verifica che f sia una matrice
        if not isinstance(f, np.ndarray):
            raise RuntimeError(
                'Input sbagliato: f deve essere una matrice NumPy quando y0 è un vettore')

        A = f
        d = len(y0)

        # Verifica che A sia quadrata e compatibile con y0
        if A.shape[0] != A.shape[1] or A.shape[0] != d:
            raise ValueError(
                "La matrice A deve essere quadrata e della stessa dimensione di y0")

        # Inizializza la matrice soluzione
        t_h = np.zeros(N+1)     # Abbiamo N passi temporali, quindi N+1 nodi
        u_h = np.zeros((d, N+1))

        # Ciclo iterativo che risolve (I-h*A)u^{n+1} = u^n
        u_h[:, 0] = y0
        t_h[0] = t0

        # Fattorizzazione LU della matrice I-h*A
        P, L, U = lu(np.eye(A.shape[0]) - h*A)

        for i in range(N):
            u_old = u_h[:, i]
            # Sostituzione in avanti e indietro per risolvere il sistema lineare
            y = fwsub(L, P.T @ u_old)
            u_h[:, i+1] = bksub(U, y)
            t_h[i+1] = t_h[i] + h

        return t_h, u_h


def crank_nicolson(f, t0, tN, y0, h, nmax_pf=300, toll_pf=1e-5):
    """
    Metodo di Crank-Nicolson.

    Risolve il problema ai valori iniziali:
        y'(t) = f(t, y(t)),  t ∈ [t0, tN]
        y(t0) = y0

    utilizzando lo schema semi-implicito (trapezoidale):
        y_{n+1} = y_n + (h/2)*(f(t_n, y_n) + f(t_{n+1}, y_{n+1}))

    Input:
        f (callable): Termine di destra dell'ODE, passato come funzione di tempo e spazio,
                     f = f(t, y)
        t0 (float): Tempo iniziale
        tN (float): Tempo finale
        y0 (float): Dato iniziale
        h (float): Passo temporale
        nmax_pf (int): Numero massimo di iterazioni per il punto fisso (default: 300)
        toll_pf (float): Tolleranza per il criterio di arresto del punto fisso (default: 1e-5)

    Output:
        t_h (numpy.ndarray): Vettore degli istanti temporali (lunghezza N+1)
        u_h (numpy.ndarray): Soluzione discreta nei nodi temporali

    Nota:
        Il metodo è di ordine 2 e incondizionatamente stabile per problemi lineari.
        Ideale per problemi non stiff che richiedono accuratezza superiore.
    """
    # Determina il numero di passi temporali N
    N = int((tN - t0) / h)

    # Inizializza la matrice soluzione
    t_h = np.zeros(N+1)     # Abbiamo N passi temporali, quindi N+1 nodi
    u_h = np.zeros(N+1)

    # Ciclo iterativo che calcola i passi
    u_h[0] = y0
    t_h[0] = t0

    for i in range(N):
        # Definisce la funzione di punto fisso per Crank-Nicolson
        def phi(z): return u_h[i] + h * \
            (f(t_h[i], u_h[i]) + f(t_h[i] + h, z)) / 2
        # Chiama il metodo del punto fisso
        u_pf = puntofisso(phi, u_h[i], nmax_pf, toll_pf)
        # Carica il vettore u
        u_h[i+1] = u_pf[-1]
        # Carica il vettore t
        t_h[i+1] = t_h[i]+h

    return t_h, u_h


def puntofisso(phi, x0, nmax=100, toll=1.0e-6):
    """
    Metodo del punto fisso (fixed-point iteration).

    Trova il punto fisso dell'equazione x = phi(x) mediante iterazione funzionale.

    Input:
        phi (callable): Funzione di iterazione
        x0 (float): Punto di partenza
        nmax (int): Numero massimo di iterazioni (default: 100)
        toll (float): Tolleranza richiesta (default: 1e-6)

    Output:
        xvect (numpy.ndarray): Vettore delle iterate

    Nota:
        Il metodo converge se |φ'(x*)| < 1 in un intorno del punto fisso x*.
    """

    # Inizializzazione
    xvect = []
    xold = x0

    for nit in range(nmax):
        # Calcola il nuovo punto
        xnew = phi(xold)
        # Accumula nel vettore
        xvect.append(xnew)

        # Criterio di arresto e aggiornamento
        if abs(xnew - xold) < toll:
            break
        else:
            xold = xnew

    return np.array(xvect)


def bksub(A, b):
    """
    Algoritmo di sostituzione all'indietro (backward substitution).

    Risolve un sistema lineare Ax = b dove A è una matrice triangolare superiore
    utilizzando l'algoritmo di sostituzione all'indietro.

    Input:
        A (numpy.ndarray): Matrice triangolare superiore quadrata
        b (numpy.ndarray): Vettore del termine noto

    Output:
        x (numpy.ndarray): Soluzione del sistema lineare Ax = b

    Eccezioni:
        RuntimeError: Se la matrice non è quadrata, non è triangolare superiore
                     o è singolare
    """

    # Dimensione del vettore b
    n = b.shape[0]

    # Verifica che la matrice sia quadrata
    if A.shape[0] != A.shape[1]:
        raise RuntimeError("ERRORE: matrice non quadrata")

    # Verifica che la matrice sia triangolare superiore
    if (A != np.triu(A)).any():
        raise RuntimeError("ERRORE: matrice non triangolare superiore")

    # Verifica che la matrice sia invertibile
    # Essendo triangolare, i suoi autovalori si trovano sulla diagonale principale
    if np.prod(np.diag(A)) == 0:
        raise RuntimeError("ERRORE: matrice singolare")

    # Inizializza il vettore x
    x = np.zeros(n)

    # x[n-1] = b[n-1]/A[n-1,n-1]
    x[-1] = b[-1] / A[-1, -1]

    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - A[i, i + 1: n] @ x[i + 1: n]) / A[i, i]

    return x


def fwsub(A, b):
    """
    Algoritmo di sostituzione in avanti (forward substitution).

    Risolve un sistema lineare Ax = b dove A è una matrice triangolare inferiore
    utilizzando l'algoritmo di sostituzione in avanti.

    Input:
        A (numpy.ndarray): Matrice triangolare inferiore quadrata
        b (numpy.ndarray): Vettore del termine noto

    Output:
        x (numpy.ndarray): Soluzione del sistema lineare Ax = b

    Eccezioni:
        RuntimeError: Se la matrice non è quadrata, non è triangolare inferiore
                     o è singolare
    """

    # Dimensione del vettore b
    n = b.shape[0]

    # Verifica che la matrice sia quadrata
    if A.shape[0] != A.shape[1]:
        raise RuntimeError("ERRORE: matrice non quadrata")

    # Verifica che la matrice sia triangolare inferiore
    if (A != np.tril(A)).any():
        raise RuntimeError("ERRORE: matrice non triangolare inferiore")

    # Verifica che la matrice sia invertibile
    # Essendo triangolare, i suoi autovalori si trovano sulla diagonale principale
    if np.prod(np.diag(A)) == 0:
        raise RuntimeError("ERRORE: matrice singolare")

    # Inizializza il vettore
    x = np.zeros(n)
    # Costruzione della sostituzione in avanti
    x[0] = b[0] / A[0, 0]

    for i in range(1, n):
        x[i] = (b[i] - A[i, 0:i] @ x[0:i]) / A[i, i]

    return x
