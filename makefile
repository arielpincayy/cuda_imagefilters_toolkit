# --- Compiladores ---
NVCC = nvcc
CC = gcc

# --- Flags ---
# -fPIC es obligatorio para crear librerías compartidas (.so)
CFLAGS = -I. -O2 -fPIC
NVCCFLAGS = -I. -O2 -std=c++11 -Xcompiler -fPIC

# --- Directorios ---
OUT_DIR = output

# --- Fuentes ---
# 1. Fuentes Comunes (Lo que usan tanto Python como tu ejecutable C)
SRCS_COMMON_C = src/utils/image.c
SRCS_COMMON_CU = src/filters/filters.cu

# 2. Fuentes solo para el ejecutable (El que tiene el int main)
SRCS_APP_CU = src/main.cu

# --- Objetos (Generación automática de nombres .o) ---
OBJS_COMMON = $(SRCS_COMMON_C:.c=.o) $(SRCS_COMMON_CU:.cu=.o)
OBJ_APP = $(SRCS_APP_CU:.cu=.o)

# --- Targets (Salidas) ---
LIB_TARGET = $(OUT_DIR)/libmycuda.so
EXE_TARGET = $(OUT_DIR)/main

# --- Regla por defecto ---
all: directorios $(LIB_TARGET) $(EXE_TARGET)

# --- Crear directorio output si no existe ---
directorios:
	mkdir -p $(OUT_DIR)

# --- 1. Librería compartida para Python (.so) ---
# IMPORTANTE: Aquí NO incluimos OBJ_APP (main.o)
$(LIB_TARGET): $(OBJS_COMMON)
	$(NVCC) -shared -o $@ $^

# --- 2. Ejecutable normal de C/CUDA ---
# Aquí SÍ incluimos todo: comunes + main
$(EXE_TARGET): $(OBJS_COMMON) $(OBJ_APP)
	$(NVCC) -o $@ $^

# --- Reglas de Compilación ---
# Compilar C a Objetos
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compilar CUDA a Objetos
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# --- Limpiar ---
clean:
	rm -f src/utils/*.o src/filters/*.o src/*.o
	rm -f $(LIB_TARGET) $(EXE_TARGET)
	# Opcional: Borrar carpeta output si está vacía
	# rmdir $(OUT_DIR) 2>/dev/null || true