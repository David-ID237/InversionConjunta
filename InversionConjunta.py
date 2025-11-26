#!/usr/bin/env python3
"""
Inversión magnética + gravimétrica con Algoritmo Genético

Este programa realiza una inversión conjunta usando:
- Dipolos magnéticos para modelar la anomalía magnética residual
- Esferas enterradas para modelar la anomalía gravimétrica residual

Cada fuente está definida por 4 parámetros:
    x    -> posición horizontal (m)
    z    -> profundidad (m)
    m    -> momento magnético
    rho  -> densidad de la esfera

El algoritmo:
1. Carga perfiles magnéticos y gravimétricos con muestreo distinto.
2. Define un modelo directo para cada física (dipolos y esferas).
3. Calcula un misfit combinado normalizado por amplitud.
4. Ejecuta un Algoritmo Genético con selección por torneo,
   elitismo, cruzamiento uniforme y mutación.
5. Regresa la mejor solución encontrada.
6. Grafica los resultados.

Requiere:
- numpy
- matplotlib
"""

# ================================================
#               IMPORTACIONES
# ================================================
import numpy as np
import matplotlib.pyplot as plt
import random

# ================================================
#               CONFIGURACIÓN GENERAL
# ================================================
"""
Aquí se definen:
- Archivos de entrada
- Parámetros del campo geomagnético
- Número y rango de parámetros de las fuentes
- Pesos del misfit conjunto
- Parámetros del Algoritmo Genético
"""

MAG_FILE = "Magne_residual.csv"
GRAV_FILE = "Gravi_residual.csv"

SEED = 869
random.seed(SEED)
rng = np.random.default_rng(SEED)

# Inclinación, declinación y permeabilidad magnética del vacío
INC_DEG = 49.9061
DEC_DEG = 5.2704
MU0 = 4 * np.pi * 1e-7

# Número de fuentes (dipolos + esferas)
N_SOURCES = 6

# Parámetros físicos de la esfera para gravedad
RADIUS_ESFERA = 30.0
VOLUMEN_ESF = (4.0/3.0) * np.pi * (RADIUS_ESFERA**3)

# Límites de búsqueda para los parámetros
X_MIN, X_MAX = None, None  # Se asignan automáticamente desde los datos
Z_MIN, Z_MAX = 5.0, 150.0
M_MIN, M_MAX = -5e5, 5e5
RHO_MIN, RHO_MAX = 500.0, 3500.0

# Pesos del misfit (importante para balancear mag vs grav)
W_MAG = 1.0
W_GRAV = 1.5

# Configuración del Algoritmo Genético
POP_SIZE = 200
GENERATIONS = 1000
MUTATION_RATE = 0.15
ELITISM_FRACTION = 0.1
TOURNAMENT_K = 3

# Densidad media del terreno
RHO_MEAN = 1800.0

# ================================================
#               LECTURA DE DATOS
# ================================================
def load_csv_simple(fname):
    """
    Carga un CSV con dos columnas:
        x , dato_observado
    Retorna:
        xs : arreglo de posiciones
        ys : arreglo de valores observados
    """
    arr = np.genfromtxt(fname, delimiter=",", skip_header=1)
    xs = arr[:, 0].astype(float)
    ys = arr[:, 1].astype(float)
    return xs, ys

# Cargar perfiles
x_mag, T_obs = load_csv_simple(MAG_FILE)
x_grav, G_obs = load_csv_simple(GRAV_FILE)

# Definir automáticamente el rango horizontal
if X_MIN is None:
    X_MIN = float(min(np.min(x_mag), np.min(x_grav)))
if X_MAX is None:
    X_MAX = float(max(np.max(x_mag), np.max(x_grav)))

# ================================================
#            MODELOS DIRECTOS (FORWARDS)
# ================================================

def dipole_field_at_points(x_obs, x_d, z_d, moment, inc_deg=INC_DEG, dec_deg=DEC_DEG):
    """
    Calcula la anomalía total-field (nT) debida a un dipolo magnético enterrado.
    Se usa la aproximación clásica del campo de dipolo en 3D.

    Entradas:
        x_obs : posiciones del perfil
        x_d   : posición horizontal del dipolo
        z_d   : profundidad del dipolo
        moment: momento magnético escalar
        inc_deg, dec_deg : campo principal terrestre

    Salida:
        B_proj(nT) : perfil magnético modelado
    """
    rx = x_obs - x_d
    rz = -z_d * np.ones_like(rx)

    # vector distancia 3D (x, y, z)
    r_vec = np.vstack((rx, np.zeros_like(rx), rz))
    r = np.linalg.norm(r_vec, axis=0)
    r = np.where(r < 1.0, 1.0, r)  # evita singularidad
    r_hat = r_vec / r

    # dirección del campo principal F
    inc = np.deg2rad(inc_deg)
    dec = np.deg2rad(dec_deg)
    F = np.array([np.cos(inc)*np.cos(dec),
                  np.cos(inc)*np.sin(dec),
                  np.sin(inc)])

    # dipolo alineado con F
    m_vec = moment * F
    m_dot_rhat = np.sum(m_vec[:, None] * r_hat, axis=0)

    # fórmula campo dipolar
    B_vec = (MU0/(4*np.pi)) * (3*r_hat*m_dot_rhat - m_vec[:, None]) / (r**3)
    B_proj = np.sum(B_vec * F[:, None], axis=0)
    return B_proj * 1e9  # convertir a nT

def model_from_dipoles_mag(x_obs, params):
    """
    Suma los campos de todos los dipolos definidos en 'params'.
    params = [x,z,m,rho, x,z,m,rho, ...]
    Solo usa x,z,m.
    """
    N = params.size // 4
    model = np.zeros_like(x_obs)
    for i in range(N):
        xi = params[4*i + 0]
        zi = params[4*i + 1]
        mi = params[4*i + 2]
        model += dipole_field_at_points(x_obs, xi, zi, mi)
    return model

def model_from_spheres_grav(x_obs, params, rho_mean=RHO_MEAN):
    """
    Calcula la anomalía de gravedad usando esferas:
        gz = 4e-6 * pi * R^3 * (rho - rho_mean) * z / r^3
    donde r = sqrt((x - x0)^2 + z^2).

    Usa densidad individual por fuente.
    """
    N = params.size // 4
    model = np.zeros_like(x_obs)
    for i in range(N):
        xi = params[4*i + 0]
        zi = params[4*i + 1]
        rhoi = params[4*i + 3]

        dx = x_obs - xi
        r = np.sqrt(dx**2 + zi**2)
        r = np.where(r < 1.0, 1.0, r)

        delta_rho = rhoi - rho_mean
        gz = 4e-6 * np.pi * (RADIUS_ESFERA**3) * delta_rho * (zi) / (r**3)
        model += gz
    return model

# ================================================
#                FUNCIÓN DE MISFIT
# ================================================
def misfit_combined(params, x_mag, T_obs, x_grav, G_obs, w_mag=W_MAG, w_grav=W_GRAV):
    """
    Calcula el misfit combinado:
        misfit = w_mag * mis_mag + w_grav * mis_grav

    Cada misfit se normaliza por la amplitud total (ptp)
    para evitar que la gravedad domine por escala.
    """
    # --- Magnético
    T_model = model_from_dipoles_mag(x_mag, params)
    res_mag = T_obs - T_model
    s_mag = np.ptp(T_obs) if np.ptp(T_obs) != 0 else np.std(T_obs)
    mis_mag = np.sum(res_mag**2) / (s_mag**2)

    # --- Gravimétrico
    G_model = model_from_spheres_grav(x_grav, params, rho_mean=RHO_MEAN)
    res_grav = G_obs - G_model
    s_grav = np.ptp(G_obs) if np.ptp(G_obs) != 0 else np.std(G_obs)
    mis_grav = np.sum(res_grav**2) / (s_grav**2)

    return w_mag*mis_mag + w_grav*mis_grav

# ================================================
#        FUNCIONES DEL ALGORITMO GENÉTICO
# ================================================
def random_individual(N, x_bounds, z_bounds, m_bounds, rho_bounds):
    """
    Crea un individuo aleatorio dentro de los rangos.
    Cada fuente tiene 4 genes.
    """
    vec = np.zeros(4*N)
    for i in range(N):
        vec[4*i + 0] = rng.uniform(*x_bounds)
        vec[4*i + 1] = rng.uniform(*z_bounds)
        vec[4*i + 2] = rng.uniform(*m_bounds)
        vec[4*i + 3] = rng.uniform(*rho_bounds)
    return vec

def mutate_individual(ind, x_bounds, z_bounds, m_bounds, rho_bounds, rate=MUTATION_RATE):
    """
    Mutación simple por reemplazo:
    cada gen tiene probabilidad 'rate' de ser sustituido
    por otro valor aleatorio dentro de su rango.
    """
    new = ind.copy()
    for j in range(new.size):
        if rng.random() < rate:
            mod = j % 4
            if mod == 0: new[j] = rng.uniform(*x_bounds)
            elif mod == 1: new[j] = rng.uniform(*z_bounds)
            elif mod == 2: new[j] = rng.uniform(*m_bounds)
            else: new[j] = rng.uniform(*rho_bounds)
    return new

def crossover(a, b):
    """
    Cruzamiento uniforme por bloques de 4 genes.
    Cada fuente se intercambia con probabilidad 0.5.
    """
    Nblocks = a.size // 4
    child = a.copy()
    for i in range(Nblocks):
        if rng.random() < 0.5:
            child[4*i:4*i+4] = b[4*i:4*i+4]
    return child

# ================================================
#          BUCLE PRINCIPAL DEL GA
# ================================================
def genetic_inversion_combined(x_mag, T_obs, x_grav, G_obs,
                               N=N_SOURCES,
                               pop_size=POP_SIZE,
                               generations=GENERATIONS,
                               x_bounds=(X_MIN, X_MAX),
                               z_bounds=(Z_MIN, Z_MAX),
                               m_bounds=(M_MIN, M_MAX),
                               rho_bounds=(RHO_MIN, RHO_MAX),
                               elitism_fraction=ELITISM_FRACTION,
                               tournament_k=TOURNAMENT_K,
                               seed=SEED):
    """
    Implementación completa del Algoritmo Genético para optimizar la inversión.
    Incluye:
        - Inicialización aleatoria
        - Ordenamiento por fitness
        - Elitismo
        - Selección por torneo
        - Crossover y mutación
        - Historial del mejor misfit
    """
    # Inicializar población
    pop = [random_individual(N, x_bounds, z_bounds, m_bounds, rho_bounds)
           for _ in range(pop_size)]

    best_hist = []
    best_ind = None
    best_score = 1e300
    n_elite = max(1, int(pop_size * elitism_fraction))

    for g in range(generations):

        # Evaluar población
        scores = np.array([misfit_combined(ind, x_mag, T_obs, x_grav, G_obs)
                           for ind in pop])

        # Ordenar por fitness
        idx = np.argsort(scores)
        pop = [pop[i] for i in idx]
        scores = scores[idx]

        # Guardar mejor solución histórica
        if scores[0] < best_score:
            best_score = scores[0]
            best_ind = pop[0].copy()

        best_hist.append(best_score)

        if g % 10 == 0:
            print(f"Gen {g+1}/{generations} | Mejor misfit = {best_score:.6e}")

        # --- Elitismo
        new_pop = pop[:n_elite]

        # --- Rellenar resto mediante torneo+crossover+mutación
        while len(new_pop) < pop_size:

            # Selección torneo para padre 1
            cand = rng.integers(0, pop_size, size=tournament_k)
            p1 = pop[cand[np.argmin(scores[cand])]]

            # Selección torneo para padre 2
            cand = rng.integers(0, pop_size, size=tournament_k)
            p2 = pop[cand[np.argmin(scores[cand])]]

            child = crossover(p1, p2)
            child = mutate_individual(child, x_bounds, z_bounds, m_bounds, rho_bounds)
            new_pop.append(child)

        pop = new_pop

    return best_ind, np.array(best_hist)

# ================================================
#          EJECUCIÓN DE LA INVERSIÓN
# ================================================
print("Iniciando inversión combinada MAG+GRAV con seed =", SEED)

best_params, history = genetic_inversion_combined(x_mag, T_obs, x_grav, G_obs)

# Imprimir parámetros invertidos por fuente
print("\n===== Parámetros invertidos =====")
for i in range(N_SOURCES):
    xi   = best_params[4*i + 0]
    zi   = best_params[4*i + 1]
    mi   = best_params[4*i + 2]
    rhoi = best_params[4*i + 3]
    print(f"Fuente {i+1}: x={xi:.2f}, z={zi:.1f}, m={mi:.3e}, rho={rhoi:.1f}")

# ================================================
#         GRAFICAR RESULTADOS
# ================================================
T_model = model_from_dipoles_mag(x_mag, best_params)
G_model = model_from_spheres_grav(x_grav, best_params, rho_mean=RHO_MEAN)

# --- Ajuste magnético ---
plt.figure(figsize=(10,5))
plt.plot(x_mag, T_obs, '-k', label="Mag Observado")
plt.plot(x_mag, T_model, '-b', label="Mag Modelo")
plt.grid(); plt.legend()
plt.title("Ajuste magnético")

# --- Ajuste gravimétrico ---
plt.figure(figsize=(10,5))
plt.plot(x_grav, G_obs, '-k', label="Grav Observado")
plt.plot(x_grav, G_model, '-r', label="Grav Modelo")
plt.grid(); plt.legend()
plt.title("Ajuste gravimétrico")

# --- Evolución del misfit ---
plt.figure(figsize=(8,4))
plt.plot(history)
plt.yscale("log")
plt.grid()
plt.title("Evolución del misfit")

# --- Visualización de fuentes ---
xs   = best_params[0::4]
zs   = best_params[1::4]
ms   = best_params[2::4]
rhos = best_params[3::4]

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,8), sharex=True)

ax1.plot(x_grav, G_obs, '-k')
ax1.plot(x_grav, G_model, '-b')
ax1.set_title("Grav")

ax2.plot(x_mag, T_obs, '-k')
ax2.plot(x_mag, T_model, '-r')
ax2.set_title("Mag")

sizes = 50 + 1000 * (np.abs(ms) / (np.max(np.abs(ms))+1e-9))
sc = ax3.scatter(xs, zs, s=sizes, c=rhos, cmap='plasma', edgecolors='k')
ax3.invert_yaxis()
ax3.set_title("Modelo de fuentes")
plt.colorbar(sc, ax=ax3)

plt.tight_layout()
plt.show()
