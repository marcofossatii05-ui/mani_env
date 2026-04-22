"""
Modulo risolutore per leggi di conservazione 1D.

Implementa il metodo ai volumi finiti (Finite Volume Method) per la risoluzione
numerica di problemi di trasporto con leggi di conservazione della forma:
    ∂u/∂t + ∂f(u)/∂x = 0

Con supporto per diversi schemi di flusso (upwind, Godunov).

Funzioni principali:
    - fv_solve: risolutore ai volumi finiti 1D
"""

import numpy as np


def fv_solve(u0, f, df, L, T, h, dt, flux_function):
    """
    Risolve un problema di trasporto con legge di conservazione usando il metodo ai volumi finiti 1D.

    Il metodo ai volumi finiti è una tecnica robusta per la risoluzione di equazioni
    di conservazione: l'intervallo spaziale viene suddiviso in celle e si discretizzano
    i flussi alle interfacce tra celle.
    
    Equazione: ∂u/∂t + ∂f(u)/∂x = 0
    Discretizzazione: u_i^{n+1} = u_i^n - (dt/h) * (F_{i+1/2}^n - F_{i-1/2}^n)
    
    Input:
        u0 (callable): Dato iniziale - funzione u(x) al tempo t=0
        f (callable): Funzione di flusso f(u)
        df (callable): Derivata del flusso f'(u)
        L (float): Lunghezza dell'intervallo spaziale [0, L]
        T (float): Tempo finale di integrazione
        h (float): Larghezza delle celle spaziali
        dt (float): Passo temporale
        flux_function (callable): Funzione per il calcolo del flusso numerico
                                 (es. upwind_flux o godunov_flux)

    Output:
        xc (numpy.ndarray): Baricentri delle celle spaziali (vettore di lunghezza ncells)
        t (numpy.ndarray): Tempi d'evoluzione (vettore di lunghezza nt)
        u (numpy.ndarray): Approssimazione della soluzione (matrice ncells × nt)
                          Convenzione: u[i,j] approssima u(x_i, t_j)
    
    Nota:
        - La soluzione viene estesa ai bordi usando il dato iniziale
        - Il metodo è conservativo e adatto per problemi non lisci (shock)
        - Stabilità richiede condition CFL: dt/h * max|f'(u)| ≤ 1
    """

    # Costruzione delle griglie spaziali e temporali
    ncells = int(np.ceil(L / h))        # Numero di celle spaziali
    nt = int(np.ceil(T / dt) + 1)      # Numero di nodi temporali
    
    # Griglia spaziale: nodi della griglia
    x = np.linspace(0, L, ncells + 1)
    
    # Nodi sinistri e destri di ogni cella
    xL = x[0:-1]
    xR = x[1:]
    
    # Baricentri (centri) delle celle
    xc = (xL + xR) / 2.0

    # Griglia temporale
    t = np.linspace(0, T, nt)

    # Inizializzazione della matrice soluzione
    u = np.zeros((ncells, nt))
    
    # Imposizione del dato iniziale al tempo t=0
    u[:, 0] = u0(xc)

    # Ciclo temporale: integrazione da t_n a t_{n+1}
    for n in range(nt - 1):
        # Costruzione della soluzione estesa (aggiungendo valori al contorno)
        # Estensione sinistra: u0(x[0]) al bordo sinistro
        uex = np.append([u0(x[0])], u[:, n])
        # Estensione destra: u(-1,n) al bordo destro
        uex = np.append(uex, u[-1, n])

        # Calcolo dei flussi numerici alle interfacce
        # Flusso all'interfaccia tra cella i e i+1
        flusso1 = flux_function(f, df, uex[0:-2], uex[1:-1])
        flusso2 = flux_function(f, df, uex[1:-1], uex[2:])

        # Aggiornamento della soluzione tramite il metodo ai volumi finiti
        # u_i^{n+1} = u_i^n - (dt/h) * (F_{i+1/2} - F_{i-1/2})
        u[:, n + 1] = u[:, n] + (dt / h) * (flusso1 - flusso2)

    return xc, t, u
