import re
import matplotlib.pyplot as plt

# Archivo con los resultados
filename = "results.txt"

# Listas para almacenar los tiempos
rgb2gray_times = []
sobel_times = []
gauss_times = []

# Expresiones regulares para extraer valores numéricos
rgb_pattern = r"\[GPU CUDA rgb2gray\].*?([\d\.]+)\s*ms"
sobel_pattern = r"\[GPU CUDA Sobel\].*?([\d\.]+)\s*ms"
gauss_pattern = r"\[GPU CUDA GaussianBlur\].*?([\d\.]+)\s*ms"

with open(filename, "r") as file:
    content = file.read()

    rgb2gray_times = [float(t) for t in re.findall(rgb_pattern, content)]
    sobel_times    = [float(t) for t in re.findall(sobel_pattern, content)]
    gauss_times    = [float(t) for t in re.findall(gauss_pattern, content)]

# Crear eje X según cantidad de mediciones
x = list(range(1, len(rgb2gray_times) + 1))

# Graficar
plt.figure(figsize=(10, 6))

plt.plot(x, rgb2gray_times, marker='o', label="GPU CUDA rgb2gray")
plt.plot(x, sobel_times, marker='o', label="GPU CUDA Sobel")
plt.plot(x, gauss_times, marker='o', label="GPU CUDA GaussianBlur")

plt.xlabel("Escenario / Tamaño de imagen")
plt.ylabel("Tiempo (ms)")
plt.title("Tiempos de ejecución GPU CUDA a partir de results.txt")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("results_graph.png")