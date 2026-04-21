"""
Modulo per la visualizzazione di metodi numerici di integrazione.

Fornisce funzioni per disegnare e visualizzare il funzionamento di diverse
formule di quadratura numerica applicate a una funzione di prova.

Funzioni principali:
    - plot_function: disegna la funzione di prova
    - show: visualizza il metodo di integrazione scelto

Funzione di prova: f(x) = x*(sin(2x) + sin(6x)) + x definita su [0, 1]
"""

import matplotlib.pyplot as plt
from numpy import linspace, sin, array, polyfit, polyval

# Griglia di valori x per il plotting della funzione
xplot = linspace(0, 1, 1001)


def f(x):
    """
    Funzione di prova per i metodi di integrazione numerica.

    f(x) = x*(sin(2x) + sin(6x)) + x

    Input:
        x (float o numpy.ndarray): Punto(i) di valutazione

    Output:
        float o numpy.ndarray: Valore della funzione
    """
    return x * (sin(2*x) + sin(6*x)) + x


def plot_function():
    """
    Plot della funzione di prova f(x) = x*(sin(2x) + sin(6x)) + x.

    Crea un grafico riempito che mostra la funzione di prova sull'intervallo [0, 1].
    """
    plt.figure(figsize=(5, 4))
    plt.plot(xplot, f(xplot))
    plt.fill_between(xplot, 0*xplot, f(xplot), color='steelblue', alpha=0.1)
    plt.text(0.13, 0.8, '$f(x)$', fontsize=12)


def show(method, intervals=6):
    """
    Visualizza un metodo di integrazione numerica applicato alla funzione di prova.

    Disegna la funzione di prova suddivisa in sottointervalli e mostra l'approssimazione
    del metodo scelto per il calcolo dell'integrale.

    Input:
        method (str): Metodo di integrazione da visualizzare. Opzioni valide:
                     - 'Punto medio': Formula del punto medio composita
                     - 'Trapezi': Formula dei trapezi composita
                     - 'Simpson': Formula di Simpson composita
        intervals (int): Numero di sottointervalli da usare (default: 6)

    Output:
        Nessuno (visualizza il grafico della funzione con il metodo applicato)

    Eccezioni:
        RuntimeError: Se il nome del metodo non è riconosciuto

    Nota:
        Il metodo disegna l'approssimazione in rosso sovrapposta alla funzione originale
        (che è in blu) per permettere un confronto visivo.
    """

    # Creazione della griglia di intervalli
    xgrid = linspace(0, 1, intervals+1)

    # Disegno della funzione di prova
    plot_function()

    # Iterazione su ogni sottointervallo per applicare il metodo scelto
    for j in range(len(xgrid)-1):
        if (method == 'Punto medio'):
            # Formula del punto medio: rettangolo col valore nel punto medio
            # Calcolo del punto medio dell'intervallo
            x_medio = (xgrid[j] + xgrid[j+1]) * 0.5
            f_medio = f(x_medio)

            # Disegno della linea orizzontale all'altezza f(x_medio)
            plt.plot(xgrid[[j, j+1]], 2*[f_medio], '-r')

            # Riempimento dell'area sotto la linea orizzontale
            plt.fill_between(xgrid[[j, j+1]], [0, 0],
                             2*[f_medio], color='red', alpha=0.1)

        elif (method == 'Trapezi'):
            # Formula dei trapezi: retta che connette f(x_j) e f(x_{j+1})
            # Disegno della retta congiungente gli estremi
            plt.plot(xgrid[[j, j+1]], f(xgrid[[j, j+1]]), '-r')

            # Riempimento dell'area sottesa dal trapezio
            plt.fill_between(xgrid[[j, j+1]], [0, 0],
                             f(xgrid[[j, j+1]]), color='red', alpha=0.1)

        elif (method == 'Simpson'):
            # Formula di Simpson: parabola passante per gli estremi e il punto medio
            xl, xr = xgrid[[j, j+1]]
            xm = 0.5 * (xl + xr)

            # Punti per l'interpolazione: estremi sinistro, medio e destro
            xx = array([xl, xm, xr])

            # Griglia fine per il disegno della parabola
            minix = linspace(xl, xr, 1000)

            # Calcolo della parabola interpolante (polinomio di grado 2)
            miniy = polyval(polyfit(xx, f(xx), deg=2), minix)

            # Disegno della parabola
            plt.plot(minix, miniy, '-r')

            # Riempimento dell'area sotto la parabola
            plt.fill_between(minix, 0*minix, miniy, color='red', alpha=0.1)
        else:
            raise RuntimeError("Metodo sconosciuto.")

    # Visualizzazione del grafico
    plt.show()
