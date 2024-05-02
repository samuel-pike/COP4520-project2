/* ==================================================================
	Programmer: Samuel Pike U36781461
	SDH algorithm implementation adapting code written for course
	to run on GPU and compare to execution on CPU.
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>


#define BOX_SIZE 23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */
int block_size;

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}
/*
	SDH solution on GPU with one thread per point
*/
__global__ void PDH_cuda(atom * a_list, bucket * gpu_histogram, int PDH_acnt, double PDH_res, int num_buckets) {
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ bucket l_histogram[]; // Makes a shared histogram between the threads
	
	atom cur_atom = a_list[tx]; // Stores the atom for current thread
    
	// Iterates over every thread between blocks and calculates the distance, adds it to the appropriate spot in histogram
	if(tx < PDH_acnt){
		for(int j = tx + 1; j < PDH_acnt; j++) {
			double x1 = cur_atom.x_pos;
			double x2 = a_list[j].x_pos;
			double y1 = cur_atom.y_pos;
			double y2 = a_list[j].y_pos;
			double z1 = cur_atom.z_pos;
			double z2 = a_list[j].z_pos;
			double dist = sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

			int h_pos = (int) (dist / PDH_res);

			atomicAdd((int *)&(l_histogram[h_pos].d_cnt), 1); // Increments with atomic operation so memory is consistent
		}
	}

	__syncthreads();
	// Loops over the shared histogram and adds it to the output histogram
	// #pragma unroll /* unrolled loop because it only iterates < 3 times*/
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        atomicAdd((unsigned long long *)&(gpu_histogram[i].d_cnt), l_histogram[i].d_cnt);
    }

	__syncthreads();
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld s\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(bucket * myHist){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", myHist[i].d_cnt);
		total_cnt += myHist[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("|\nT:%lld \n", total_cnt);
		else printf("| ");
	}
}

void compare_histograms(bucket * myHist1, bucket * myHist2){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", myHist1[i].d_cnt - myHist2[i].d_cnt);
		total_cnt += myHist1[i].d_cnt - myHist2[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("|\nT:%lld \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
    block_size = atoi(argv[3]);


	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	double cpu_time = report_running_time();
	
	/* print out the histogram */
	output_histogram(histogram);


	dim3 blockDim(block_size, 1, 1); // Defines # threads per block
	dim3 gridDim((PDH_acnt + blockDim.x - 1) / blockDim.x);
    size_t shared_size = num_buckets * block_size; // creates a shared memory for the GPU local histogram

	bucket * gpu_histogram; // Intializes and allocates memory for host-side histogram
	gpu_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

	bucket * d_gpu_histogram; // Initializes, allocates memory for device-side histogram, and copies it to device
	cudaMalloc((void**)&d_gpu_histogram, sizeof(bucket) * num_buckets);
	cudaMemcpy(d_gpu_histogram, gpu_histogram, sizeof(bucket) * num_buckets, cudaMemcpyHostToDevice);

	atom * d_atom_list; // Initializes, allocates memory for device-side atom list, and copies it to device
	cudaMalloc((void**)&d_atom_list, sizeof(atom) * PDH_acnt);
	cudaMemcpy(d_atom_list, atom_list, sizeof(atom) * PDH_acnt, cudaMemcpyHostToDevice);

    // Start GPU timing
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	PDH_cuda<<<gridDim, blockDim, shared_size>>>(d_atom_list, d_gpu_histogram, PDH_acnt, PDH_res, num_buckets); // Calls CUDA function
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    // End GPU timing


	cudaMemcpy(gpu_histogram, d_gpu_histogram, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost); // Copies device histogram back to host

	output_histogram(gpu_histogram); // Prints the histogram as run on GPU

	compare_histograms(histogram, gpu_histogram); // Compares histograms as: histogram - gpu_histogram

    printf("********* Total Running Time of Kernel: %0.5f ms *********\n", elapsedTime);

	cudaFree(d_gpu_histogram); // Frees memory from GPU
	cudaFree(d_atom_list);
    free(histogram);
    free(atom_list);
    free(gpu_histogram);

	return 0;
}