__global__ void im2col_cuda( double * out, const double * in1 ,const int * in2) 
{
    int idy = blockDim.x*blockIdx.x + threadIdx.x;
	int idx = blockDim.y*blockIdx.y + threadIdx.y;
	int idz = blockDim.z*blockIdx.z + threadIdx.z;
	int x = idx/(in2[0]-in2[2]+1);
	int y = idx%(in2[0]-in2[2]+1);
	
	int px = idy/in2[2];
	int py = idy%in2[2];
	
    out[idx * (in2[2]*in2[3]) + idy + idz*in2[4]] = in1[(x+px)* in2[0] +(y+py)+idz*in2[0]*in2[1]];
	//out[idx * (in2[2]*in2[3]) + idy + idz*in2[4]]  = idz;
}