import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def main():
    """
    Script para ajuste de una tendencia polinomial mediante
    Mínimos Cuadrados (polyfit) y obtención de:
        - Tendencia regional
        - Serie residual
        - Gráficas automáticas
        - Archivos CSV con la regional y el residual

    Uso:
        python script.py -i datos.csv -o salida -d 2
    """

    # ================================================================
    # 1. CONFIGURACIÓN DEL PARSER DE ARGUMENTOS
    # ================================================================
    # argparse permite ejecutar el script desde consola indicando:
    #  - archivo de entrada CSV
    #  - nombre base para archivos de salida
    #  - grado del polinomio que se ajustará
    parser = argparse.ArgumentParser(
        description='Ajuste de tendencia polinomial por Mínimos Cuadrados (MMC).'
    )

    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Ruta del archivo CSV de entrada (debe tener columnas de distancia y valor).')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Nombre base para los archivos de salida (sin extensión).')
    parser.add_argument('-d', '--degree', required=True, type=int,
                        help='Grado del polinomio a ajustar.')

    args = parser.parse_args()

    # ================================================================
    # 2. CARGA Y VALIDACIÓN DE DATOS
    # ================================================================
    # Verificamos que el archivo exista antes de intentar leerlo
    if not os.path.exists(args.input):
        print(f"Error: El archivo '{args.input}' no existe.")
        sys.exit(1)

    try:
        # Carga del archivo CSV.
        # Se asume que:
        #   columna 0 → distancia o posición (x)
        #   columna 1 → variable medida (y)
        df = pd.read_csv(args.input)

        # Validación: al menos 2 columnas
        if df.shape[1] < 2:
            print("Error: El CSV debe tener al menos dos columnas.")
            sys.exit(1)

        # Extraemos los arreglos numéricos
        x_original = df.iloc[:, 0].values
        y_original = df.iloc[:, 1].values

        # Guardamos los nombres de columna para usarlos en los ejes
        x_label = df.columns[0]
        y_label = df.columns[1]

    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        sys.exit(1)

    # ================================================================
       # 3. PROCESAMIENTO MATEMÁTICO — AJUSTE POLINOMIAL (MMC)
    # ================================================================
    # Se desplaza el eje X restando su primer valor.
    #   Esto mejora la estabilidad numérica al usar polinomios
    #   (técnica común en MATLAB y métodos de MMC).
    x_shift = x_original - x_original[0]

    print(f"Calculando ajuste polinomial de grado {args.degree}...")

    # Ajuste por mínimos cuadrados:
    #   np.polyfit devuelve los coeficientes del polinomio
    coeffs = np.polyfit(x_shift, y_original, args.degree)

    # Evaluamos el polinomio para obtener la tendencia (regional)
    trend = np.polyval(coeffs, x_shift)

    # Residual = señal original - regional
    residual = y_original - trend

    # ================================================================
    # 4. GRÁFICAS DE RESULTADOS
    # ================================================================
    # Tres subgráficas:
    #   1) Señal original
    #   2) Ajuste polinomial (regional)
    #   3) Residual
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # ---------------------------
    # Gráfica 1: Señal original
    # ---------------------------
    axs[0].plot(x_original, y_original, '-k.', alpha=0.5, label='Señal Original')
    axs[0].set_ylabel(y_label)
    axs[0].set_xlabel(f'{x_label}')
    axs[0].set_title(f'{y_label} VS {x_label}')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # ---------------------------
    # Gráfica 2: Tendencia polinomial
    # ---------------------------
    axs[1].plot(x_original, y_original, '-k.', alpha=0.5, label='Señal Original')
    axs[1].plot(x_original, trend, 'r-', linewidth=2,
                label=f'Polinomio Grado {args.degree}')
    axs[1].set_ylabel('Regional [m]')
    axs[1].set_xlabel(x_label)
    axs[1].set_title(f'Ajuste Regional (Grado {args.degree})')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # ---------------------------
    # Gráfica 3: Residual
    # ---------------------------
    axs[2].plot(x_original, residual, 'b-', linewidth=1)
    axs[2].set_ylabel('Residual [m]')
    axs[2].set_xlabel(x_label)
    axs[2].set_title('Serie Residual')
    axs[2].axhline(0, color='black', linewidth=0.8)
    axs[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{args.output}_grafica.png")

    # ================================================================
    # 5. GUARDADO DE RESULTADOS EN ARCHIVOS
    # ================================================================
    # Nombres de salida
    out_regional = f"{args.output}_regional.csv"
    out_residual = f"{args.output}_residual.csv"

    # Construimos DataFrames para exportar
    df_regional = pd.DataFrame({
        x_label: x_original,
        'Regional_Fit': trend
    })

    df_residual = pd.DataFrame({
        x_label: x_original,
        'Residual': residual
    })

    try:
        df_regional.to_csv(out_regional, index=False)
        df_residual.to_csv(out_residual, index=False)

        print("\nArchivos guardados exitosamente:")
        print(f" -> {out_regional}")
        print(f" -> {out_residual}")
        print(f" -> {args.output}_grafica.png")

    except Exception as e:
        print(f"Error al guardar archivos: {e}")


# ================================================================
# EJECUCIÓN DEL SCRIPT
# ================================================================
# Solo se ejecuta main() si el archivo se corre directamente.
# Esto permite importar sus funciones sin ejecutarlo.
if __name__ == "__main__":
    main()
