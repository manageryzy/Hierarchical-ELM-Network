__global__ void calct(double *out, const double *tv, const double *im, const int *param)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    int size1x = param[0];
    int size1y = param[1];
    int size2x = param[2];
    int size2y = param[3];

    double tmpSum = 0;
    for (int i = 0; i < size1y; i++)
    {
        tmpSum += tv[i * size1x + idx ] * im[idy * size2x + i + idz * size2x * size2y];
    }

    // out[idy * size1x + idx + idz * (size1x * size2y)] = tmpSum;
    out[idy * size1x + idx + idz * (size1x * size2y)] = 2/(1+exp(-2*tmpSum))-1;
}