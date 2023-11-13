#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

// Declarar las variables en memoria constante.
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// Función para calcular el tiempo transcurrido entre dos eventos CUDA.
float GetElapsedTime(cudaEvent_t start, cudaEvent_t stop) {
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

// Kernel memoria compartida
__global__ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  int locID = threadIdx.x;

  if (gloID >= w * h) return;

  int xCent = w / 2;
  int yCent = h / 2;
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  // Definir un acumulador local en memoria compartida
  __shared__ int localAcc[degreeBins * rBins];
  
  // Inicializar a 0 todos los elementos de este acumulador local
  localAcc[locID] = 0;
  __syncthreads();

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
    }
  }

  __syncthreads();

  // Usar una barrera para asegurarse de que todos los hilos hayan completado el proceso de inicialización
  __syncthreads();

  // Actualizar el acumulador global acc utilizando el acumulador local localAcc
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
  {
    atomicAdd(&acc[i], localAcc[i]);
  }

  // Usar una segunda barrera para asegurarse de que todos los hilos hayan completado el proceso de incremento
  __syncthreads();

  // Agregar un loop para sumar los valores del acumulador local localAcc al acumulador global acc
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
  {
    atomicAdd(&acc[i], localAcc[i]);
  }
}

int main(int argc, char **argv) {
    PGMImage inImg(argv[1]);

    // Crear eventos CUDA para medir el tiempo.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    // CPU_HoughTran(inImg.pixels, w, h, &cpuht); // Comentar esta línea ya que no estás usando la versión CPU

    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels;
    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int blockNum = ceil(w * h / 256);

    // Registrar el tiempo de inicio.
    cudaEventRecord(start);

    GPU_HoughTranShared<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

    // Registrar el tiempo de finalización.
    cudaEventRecord(stop);

    // Sincronizar para asegurarse de que el kernel haya terminado.
    cudaDeviceSynchronize();

    // Calcular el tiempo transcurrido.
    float elapsedTime = GetElapsedTime(start, stop);

    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < degreeBins * rBins; i++) {
    //     if (cpuht[i] != h_hough[i])
    //         printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    // }
    printf("Done!\n");

    printf("Tiempo transcurrido: %f ms\n", elapsedTime);

    // Limpieza
    free(pcCos);
    free(pcSin);
    // delete[] cpuht; // Comentar esta línea ya que no estás usando la versión CPU
    free(h_hough);
    cudaFree(d_in);
    cudaFree(d_hough);
    
    // Destruir los eventos CUDA.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
