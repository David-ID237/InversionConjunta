# InversionConjunta — Inversión magnética y gravimétrica con Algoritmo Genético

Este repositorio contiene una implementación completa para realizar inversión conjunta de anomalías magnéticas y gravimétricas. El modelo utiliza dipolos para el campo magnético residual y esferas para la anomalía gravimétrica. La optimización de parámetros se lleva a cabo mediante un Algoritmo Genético (GA). También se incluye un script independiente para obtener residuales a partir de un ajuste polinomial por Mínimos Cuadrados (MMC).

---

## Contenido del repositorio

- El archivo `InversionConjunta.py` contiene el algoritmo de inversión conjunta basado en dipolos (magnetometría) y esferas (gravimetría), optimizado con GA.
- El archivo `MCC.py` realiza el ajuste de tendencia polinomial mediante MMC y genera los residuales magnéticos y gravimétricos.
- Los archivos `data_m.csv` y `data_g.csv` contienen los perfiles originales.
- Los archivos `Magne_residual.csv` y `Gravi_residual.csv` contienen los residuales procesados.
- El archivo `requirements.txt` incluye las dependencias necesarias para ejecutar ambos scripts.

---

## Instalación

Para facilitar la ejecución del código se recomienda usar un entorno virtual de Python.

Crear entorno virtual:
```
python3 -m venv env
source env/bin/activate
```

Instalar dependencias:
```
pip install -r requirements.txt
```

---

## Uso

### 1. Obtener residuales con `MCC.py`

El script realiza un ajuste polinomial por Mínimos Cuadrados al perfil original y genera:

- El regional calculado
- El residual resultante
- Gráficas del ajuste
- Archivos CSV exportados

Ejemplo de ejecución:
```
python MCC.py -i data_m.csv -o mag -d 3
python MCC.py -i data_g.csv -o grav -d 2
```

Esto generará archivos como:
```
mag_regional.csv
mag_residual.csv
grav_regional.csv
grav_residual.csv
```

---

## 2. Ejecutar la inversión conjunta

El archivo `InversionConjunta.py` utiliza automáticamente:
```
Magne_residual.csv
Gravi_residual.csv
```

Para ejecutar el algoritmo:
```
python InversionConjunta.py
```

El script:

- Carga ambos residuales
- Modela el campo magnético con dipolos
- Modela la anomalía gravimétrica con esferas
- Evalúa el misfit normalizado
- Ejecuta un Algoritmo Genético con selección por torneo, mutación con límites y elitismo
- Produce gráficas y parámetros óptimos

---

## Modelo directo

### Magnetometría (dipolos)
Cada dipolo incluye parámetros de posición horizontal `x`, profundidad `z` y momento magnético `m`.  
Se aplican la inclinación y declinación del campo geomagnético para calcular la componente medida.

### Gravimetría (esferas)
Cada fuente se modela como una esfera con radio fijo y densidad variable.  
La contribución se calcula usando la fórmula clásica de gravedad vertical para esferas enterradas.

---

## Algoritmo Genético

El GA ajusta simultáneamente los parámetros del conjunto completo de dipolos y esferas.  
Los pesos del misfit son:

```
W_MAG = 1.0
W_GRAV = 1.5
```

El misfit total se define como:
```
Misfit = W_MAG * mis_mag + W_GRAV * mis_grav
```

Cada componente se normaliza usando el "peak-to-peak" del perfil correspondiente.

El GA incluye:
- Selección por torneo
- Cruzamiento uniforme
- Mutación con límites físicos
- Elitismo del 10%
- Población típica de 200 individuos
- Alrededor de 1000 generaciones

---

## Salidas del programa

El script genera:
- Gráfica observada vs modelada para magnetometría
- Gráfica observada vs modelada para gravimetría
- Evolución del misfit por generación
- Parámetros óptimos del modelo final mostrados en consola

Ejemplo de parámetros mostrados:
```
Dipolo 1: x = 320.4, z = 41.1, m = 2.51e5
Esfera 1: x = 312.9, z = 33.0, densidad = 2100
Misfit final = 0.94
```

---

## Notas finales

- Es importante que los residuales estén correctamente calculados; una mala remoción de tendencia afecta la convergencia.
- Los rangos físicos de densidad, profundidad y momento deben ajustarse al tipo de terreno o litología.
- El número de fuentes (dipolos y esferas) puede modificarse directamente en el código dentro del archivo principal.
