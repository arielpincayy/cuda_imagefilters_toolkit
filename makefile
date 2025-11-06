NVCC = nvcc
CC = gcc

CFLAGS = -I. -O2
NVCCFLAGS = -I. -O2 -std=c++11

SRCS_C = src/utils/image.c
SRCS_CU = src/main.cu src/filters/filters.cu

OBJS_C = $(SRCS_C:.c=.o)
OBJS_CU = $(SRCS_CU:.cu=.o)

TARGET = output/main

all: $(TARGET)

$(TARGET): $(OBJS_C) $(OBJS_CU)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS_C) $(OBJS_CU) $(TARGET)