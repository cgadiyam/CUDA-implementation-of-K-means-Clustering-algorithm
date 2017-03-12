#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <float.h>   //FLT_MAX
#include "KMeans.h"

__constant__ Vector2 Clusters[3];

__global__ void KMeansKernel( Datapoint* data, long n, int k )
{
		//Assignment of each data point to a cluster
		int threadID = (blockIdx.x * blockDim.x) + threadIdx.x;
		if(threadID < n)
		{
			float Min_Dist = FLT_MAX;
			int nearest_cluster = 0;
			data[threadID].altered = false;
			for(int j=0;j<k;j++)
			{
				if(data[threadID].p.distSq(Clusters[j]) < Min_Dist)
				{
					Min_Dist = data[threadID].p.distSq(Clusters[j]);
					nearest_cluster = j;
				}
			}
			if(nearest_cluster != data[threadID].cluster)
			{
				data[threadID].cluster = nearest_cluster;
				data[threadID].altered = true;
			}
		}
}

bool KMeansGPU( Datapoint* data, long n, Vector2* clusters, int k )
{
	cudaError_t status;
	bool exit = false;
	int count;
	Vector2 Center;
	int bytes1 = k * sizeof(Vector2);
	cudaMalloc((void**) &Clusters, bytes1);
	cudaMemcpyToSymbol(Clusters, clusters, bytes1, 0, cudaMemcpyHostToDevice);
	Datapoint* DataSet;
	int bytes2 = n * sizeof(Datapoint);
	cudaMalloc((void**) &DataSet, bytes2);
	
    //iterates until no data point changes its cluster
	while(!exit)
	{
		count = 0;
		exit = true;

		cudaMemcpy(DataSet, data, bytes2, cudaMemcpyHostToDevice);
		dim3 dimBlock(768, 1); 
		dim3 dimGrid((int)ceil((float)n/768), 1);
		KMeansKernel<<<dimGrid, dimBlock>>>(DataSet, n, k);
		// Wait for completion
		cudaThreadSynchronize();
		// Check for errors
		status = cudaGetLastError();
		if (status != cudaSuccess) 
		{
			std::cout << "Kernel failed: " << cudaGetErrorString(status) << std::endl;
			cudaFree(DataSet);
			return false;
		}
		// Retrieve the result matrix
		cudaMemcpy(data, DataSet, bytes2, cudaMemcpyDeviceToHost);

		//calculation of new center for all 3 clusters
		for(int i=0;i<k;i++)
		{
			count = 0;
			Center.x = 0;
			Center.y = 0;
			for(int j=0;j<n;j++)
			{
				if(data[j].cluster == i)
				{
					Center.x += data[j].p.x;
					Center.y += data[j].p.y;
					count++;
				}
			}
			if(count >0)
			{
				clusters[i].x = (Center.x)/count;
				clusters[i].y = (Center.y)/count;
			}
		}
		cudaMemcpyToSymbol(Clusters, clusters, bytes1, 0, cudaMemcpyHostToDevice);
		for(int i=0;i<n;i++)
		{
			if(data[i].altered == true)
			{
				data[i].altered = false;
				exit = false;
			}
		}
	}

	cudaFree(DataSet);
	// Success
	return true;
}