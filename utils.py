"""
Modulo di utilità per i metodi di risoluzione iterativi di sistemi lineari.

Implementa il metodo del gradiente a parametro dinamico (steepest descent method)
per la risoluzione di sistemi lineari Ax = b.

Funzioni principali:
    - gdescent: metodo del gradiente a parametro dinamico
"""

import numpy as np


def gdescent(A, b, x0, nmax=1000, rtoll=1e-6):
    """
    Metodo del gradiente a parametro dinamico (Steepest Descent).
    
    Risolve il sistema lineare Ax = b utilizzando il metodo del gradiente con 
    parametro di passo dinamico (ottimale).
    Applicabile a matrici simmetriche definite positive.
    
    Algoritmo:
        1. Calcola il residuo iniziale: r_0 = b - A*x_0
        2. Per k = 0, 1, ... fino a convergenza:
           - Scegli direzione di discesa: z_k = r_k (per SD, è il gradiente)
           - Calcola passo ottimale: alpha_k = (r_k, z_k) / (A*z_k, z_k)
           - Aggiorna soluzione: x_{k+1} = x_k + alpha_k * z_k
           - Aggiorna residuo: r_{k+1} = r_k - alpha_k * A*z_k
    
    Input:
        A (numpy.ndarray): Matrice del sistema simmetrica def. positiva (n×n)
        b (numpy.ndarray): Termine noto (vettore di dimensione n)
        x0 (numpy.ndarray): Vettore iniziale di innesco (dimensione n)
        nmax (int): Numero massimo di iterazioni (default: 1000)
        rtoll (float): Tolleranza relativa sul residuo per l'arresto (default: 1e-6)
                      Criterio di arresto: ||r_k|| / ||b|| < rtoll
    
    Output:
        xiter (list): Lista contenente tutte le iterate [x_0, x_1, ..., x_k]
    
    Nota:
        - Il metodo converge per matrici simmetriche definite positive
        - Velocità di convergenza lineare, dipendente dal numero di condizionamento
        - Preferibile per matrici sparse di grandi dimensioni
    """
    
    norm = np.linalg.norm

    # Norma del termine noto per il calcolo del residuo relativo
    bnorm = norm(b)

    # Residuo iniziale r_0 = b - A*x_0
    r = b - A @ x0

    # Lista delle iterate
    xiter = [x0]
    it = 0

    # Ciclo iterativo
    while (norm(r) / bnorm) > rtoll and it < nmax:
        xold = xiter[-1]

        # Direzione di discesa (per il metodo del gradiente: z = r)
        z = r
        
        # Calcolo del parametro di passo ottimale
        # alpha = (r, z) / (A*z, z)
        rho = np.dot(r, z)
        q = A @ z
        alpha = rho / np.dot(z, q)
        
        # Aggiornamento della soluzione
        xnew = xold + alpha * z
        
        # Aggiornamento del residuo
        r = r - alpha * q

        # Accumulo nel vettore delle iterate
        xiter.append(xnew)
        it = it + 1

    return xiter
