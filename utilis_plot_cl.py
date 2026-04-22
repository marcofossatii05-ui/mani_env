"""
Modulo di visualizzazione per problemi con leggi di conservazione.

Fornisce funzioni per visualizzare e animare le soluzioni numeriche di
problemi di trasporto con leggi di conservazione 1D.

Funzioni principali:
    - xtplot: visualizza la soluzione come animazione o superficie 3D
    - plot_frame: disegna un frame singolo della soluzione
    - fvplot: disegna una rappresentazione grafica della soluzione ai volumi finiti
    - save_gif: salva un'animazione come file GIF
    - animate_conservation_laws: crea un'animazione della soluzione
    - surface_conservation_laws: visualizza la soluzione come superficie 3D
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.colors import Colormap


def xtplot(x, t, u, plot_type='animation', color='blue', piecewise=True, name_gif='solution_animation'):
    """Plot della soluzione di un problema con legge di conservazione.

    Visualizza la soluzione numerica calcolata con il metodo ai volumi finiti
    per problemi di trasporto con leggi di conservazione.

    Input:
        x (numpy.ndarray): Baricentri delle celle spaziali (vettore)
        t (numpy.ndarray): Tempi d'evoluzione (vettore)
        u (numpy.ndarray): Approssimazione della soluzione (matrice ncells × nt)
                          Convenzione: u[i,j] approssima u(x_i, t_j)
        plot_type (str): Tipo di visualizzazione. Opzioni:
                        - 'animation': crea un'animazione GIF
                        - 'surface': visualizza la soluzione come superficie 3D
        color (str): Colore della visualizzazione (default: 'blue')
        piecewise (bool): Se True, disegna la soluzione come costante a tratti (default: True)
        name_gif (str): Nome del file GIF (se plot_type='animation')

    Output:
        Nessuno (visualizza il grafico o salva GIF)

    Eccezioni:
        ValueError: Se plot_type non è riconosciuto
    """
    if plot_type == 'animation':
        animate_conservation_laws(x, t, u, color, piecewise, name_gif)
    elif plot_type == 'surface':
        surface_conservation_laws(x, t, u)
    else:
        raise ValueError("Errore: il plot_type in input è sbagliato")


def plot_frame(x, t, u, i, color='blue', piecewise=True):
    """Disegna il frame i-esimo della soluzione.

    Crea un grafico della soluzione al tempo t[i] in un intervallo
    spaziale opportuno.

    Input:
        x (numpy.ndarray): Baricentri delle celle spaziali (vettore)
        t (numpy.ndarray): Tempi d'evoluzione (vettore)
        u (numpy.ndarray): Approssimazione della soluzione (matrice ncells × nt)
        i (int): Indice del frame (deve essere < len(t))
        color (str): Colore della visualizzazione (default: 'blue')
        piecewise (bool): Se True, disegna la soluzione come costante a tratti (default: True)

    Output:
        Nessuno (disegna in figura matplotlib)

    Eccezioni:
        ValueError: Se l'indice i è fuori intervallo
    """
    if i >= len(t):
        raise ValueError("Errore: indice frame fuori intervallo")

    fig, ax = plt.subplots(figsize=(6, 5))

    # Disegno della soluzione al tempo t[i]
    fvplot(x, u[:, i], color, piecewise)

    # Etichette e titolo
    ax.set(xlabel='x', ylabel='u(x,t)',
           title=f'Current time: t = {t[i]:.2f}',
           xlim=(x[0] - (x[1] - x[0]), x[-1] + (x[1] - x[0])),
           ylim=(np.min(u) - 0.3 * (np.max(u) - np.min(u)),
                 np.max(u) + 0.3 * (np.max(u) - np.min(u))))


def fvplot(x, v, c, pcw):
    """Disegna la soluzione ai volumi finiti (costante a tratti o lineare).

    Visualizza i valori della soluzione ai volumi finiti: se pcw=True,
    disegna la soluzione come funzione a gradini (costante in ogni cella),
    altrimenti come funzione lineare per cellastocolo.

    Input:
        x (numpy.ndarray): Baricentri delle celle (coordinata x)
        v (numpy.ndarray): Valori della soluzione in ogni cella
        c (str): Colore della linea
        pcw (bool): Se True, disegna costante a tratti; altrimenti lineare

    Output:
        Nessuno (modifica figura matplotlib corrente)
    """
    if (pcw):
        # Disegno costante a tratti
        h = (x[1] - x[0]) / 2  # Half-width della cella
        plt.plot([x[0] - h, x[0] + h], [v[0], v[0]], color=c)

        for i in range(1, len(x)):
            # Linea verticale che connette i due livelli
            plt.plot([x[i] - h, x[i] - h], [v[i - 1], v[i]], '--', color=c)
            # Linea orizzontale al livello v[i]
            plt.plot([x[i] - h, x[i] + h], [v[i], v[i]], color=c)

    else:
        # Disegno lineare
        plt.plot(x, v, color=c)


def save_gif(draw_frame, frames, filename, dt=1.0 / 24.0):
    """Salva un'animazione in formato GIF.

    Genera una sequenza di frame chiamando la funzione draw_frame e
    li combina in un file GIF con durata specificata per ogni frame.

    Input:
        draw_frame (callable): Funzione che disegna l'i-esimo frame.
                              Deve accettare un indice intero come argomento.
        frames (int): Numero totale di frame da generare
        filename (str): Nome del file GIF di output
        dt (float): Durata di ogni frame in secondi (default: 1/24 per 24 fps)

    Output:
        Nessuno (crea file GIF sul disco)
    """
    images = []

    # Generazione di tutti i frame
    for i in range(frames):
        draw_frame(i)
        fig = plt.gcf()
        fig.canvas.draw()
        # Converte il frame in array RGBA
        images.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close(fig)

    # Salva come GIF (rimuove .gif se già presente nel nome)
    imageio.mimsave(filename.replace(".gif", "") + ".gif", images, duration=dt)


def animate_conservation_laws(x, t, u, color='blue', piecewise=True, name='solution_animation'):
    """Crea un'animazione GIF della soluzione.

    Genera un'animazione che mostra l'evoluzione della soluzione nel tempo
    e la visualizza nel notebook Jupyter.

    Input:
        x (numpy.ndarray): Baricentri delle celle spaziali
        t (numpy.ndarray): Tempi d'evoluzione
        u (numpy.ndarray): Soluzione (matrice ncells × nt)
        color (str): Colore della visualizzazione (default: 'blue')
        piecewise (bool): Se True, disegna costante a tratti (default: True)
        name (str): Nome della GIF (senza estensione)

    Output:
        Nessuno (visualizza GIF nel notebook Jupyter)
    """
    def draw_frame(i):
        return plot_frame(x, t, u, i, color, piecewise)

    # Salva l'animazione come GIF
    save_gif(draw_frame, len(t), name)

    # Visualizza nel notebook
    from IPython.display import Image, display
    name_gif = name + '.gif'
    display(Image(name_gif))


def surface_conservation_laws(x, t, u):
    """Visualizza la soluzione come superficie 3D.

    Crea un grafico tridimensionale che mostra la soluzione come una
    superficie nello spazio (x, t, u).

    Input:
        x (numpy.ndarray): Baricentri delle celle spaziali
        t (numpy.ndarray): Tempi d'evoluzione
        u (numpy.ndarray): Soluzione (matrice ncells × nt)

    Output:
        Nessuno (visualizza il grafico 3D)
    """
    X, T = np.meshgrid(x, t, indexing='ij')

    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Prima figura: superficie 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, T, u, cmap='plasma', edgecolor='k',
                     linewidth=0.3, alpha=0.9, antialiased=True, shade=True)
    ax1.set(xlabel='x', ylabel='t', zlabel='u(x,t)', title='Surface plot')
    ax1.view_init(elev=30, azim=-90)

    # Limiti opzionali per una migliore visualizzazione (commentato)
    epsilon = 1.e-1
    ax1.set_xlim(X.min()-epsilon, X.max()+epsilon)
    ax1.set_ylim(T.min()-epsilon, T.max()+epsilon)
    ax1.set_zlim(u.min()-epsilon, u.max()+epsilon)

    # seconda figura
    ax2 = fig.add_subplot(1, 2, 2)
    pcolor = ax2.pcolor(X, T, u, cmap='plasma')
    fig.colorbar(pcolor, ax=ax2)
    ax2.set(xlabel='x', ylabel='t', title='Top view')
    ax2.set_aspect('equal')

    plt.show()
