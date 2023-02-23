#include <iostream>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <fstream>
#include <stdio.h>
#include <random>
//This program simulates restricted diffusion with exchange in a substrate of anything
//Does not save trajectories
//Only saves signal

//simulation options; will be read from an options txt file
struct options
{
	long long Npart;
	double T;
	double sim_dt;
	double samp_dt;
	long long n_dim;
	double D0;
	long long sim_Nt;
	long long save_Nt;
	double ds;
	long long N_save; //N time points x N particles
	long long N_sim;
	bool save_states; //save particle state history to file or not
	double kappa; //membrane permeability
		//additional gwf options
	long long n_acq; //# b-values,
	long long n_gwf_points; // # time points in each waveform
	double max_b_value;
	double gamma; //gyromagnetic ratio
	long long delay; //number of time steps to take before acquiring signals
	bool save_positions; //save final positions or not; useful for debugging
};


//world data
struct world
{
	long long num_voxels;
	double vox_size, max_x, max_y, max_z, x_length, y_length, z_length, f1;
};

__global__ void random_init(curandState* states)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock64()+index, index, 0, &states[index]);
}


__device__ void pair(long long x, long long y, long long z, long long& xyz)
{
	//device function for pairing particle coordinates
	long long xy;
	x >= 0 ? x = 2 * x : x = -2 * x - 1;
	y >= 0 ? y = 2 * y : y = -2 * y - 1;
	z >= 0 ? z = 2 * z : z = -2 * z - 1;

	x >= y ? xy = x * x + x + y : xy = y * y + x;
	xy >= 0 ? xy = 2 * xy : xy = -2 * xy - 1;

	xy >= z ? xyz = xy * xy + xy + z : xyz = z * z + xy;
}


__device__ void find(long long* A, long long sizeA, long long a, long long& vox_idx)
{
	//find index of element in array
	for (int c = 0; c < sizeA; c++)
	{
		if (A[c] == a) { vox_idx = c; break; }
	}
}

//we will try binarySearch here instead of the slow find above
__device__ long long binary_search_rec(long long* A, long long lower, long long upper, long long x)
{
	if (upper >= lower) {
		long long mid = lower + (upper - lower) / 2;
		if (A[mid] == x)
			return mid;
		if (A[mid] > x)
			return binary_search_rec(A, lower, mid - 1, x);
		return binary_search_rec(A, mid + 1, upper, x);
	}
	return -1;
}

//binary search without recursion to avoid potentially filling the stack
__device__ long long binary_search_iter(long long* A, long long lower, long long upper, long long x)
{
	while (upper >= lower) {
		long long mid = lower + (upper - lower) / 2;
		if (A[mid] == x) return mid;
		(A[mid] > x) ? upper = mid - 1 : lower = mid + 1;
	}
	return -1;
}


__device__ void move(double& tmp_x, double& tmp_y, double& tmp_z, double& tmp_dx,
	double& tmp_dy, double& tmp_dz, int entry, curandState* states, int index, options* opt)
{
	tmp_dx = curand_normal(&states[index]);
	tmp_dy = curand_normal(&states[index]);
	tmp_dz = curand_normal(&states[index]);

	double norm = (*opt).ds * rnorm3d(tmp_dx, tmp_dy, tmp_dz);

	tmp_dx *= norm;
	tmp_dy *= norm;
	tmp_dz *= norm;

	tmp_x += tmp_dx;
	tmp_y += tmp_dy;
	tmp_z += tmp_dz;
}

__device__ void restrict_to_world(double& e_x, double& e_y, double& e_z, options* opt, world* w, double& tmp_x, double& tmp_y, double& tmp_z)
{
	if (tmp_x < -w->max_x) { tmp_x += w->x_length; e_x -= w->x_length; }
	if (tmp_x >= w->max_x) { tmp_x -= w->x_length; e_x += w->x_length; }

	if (tmp_y < -w->max_y) { tmp_y += w->y_length; e_y -= w->y_length; }
	if (tmp_y >= w->max_y) { tmp_y -= w->y_length; e_y += w->y_length; }

	if (tmp_z < -w->max_z) { tmp_z += w->z_length; e_z -= w->z_length; }
	if (tmp_z >= w->max_z) { tmp_z -= w->z_length; e_z += w->z_length; }
}


__device__ void check_state(world* w, options* opt, double& tmp_x, double& tmp_y, double& tmp_z, double& tmp_dx,
	double& tmp_dy, double& tmp_dz, long long& tmp_loc, long long* table, curandState* states, int index, long long& vox_idx)
{
	bool reject = false; //reject move or not
	//first, identify which voxel the particle is in
	long long x_pos = floor(tmp_x / w->vox_size);
	long long y_pos = floor(tmp_y / w->vox_size);
	long long z_pos = floor(tmp_z / w->vox_size);
	double p_ex, p_12, p_21;
	long long xyz, old_vox_idx; //voxel identifier
	old_vox_idx = vox_idx; //save curent identifier in case particle is rejected after move

	pair(x_pos, y_pos, z_pos, xyz); //get the identifier

	vox_idx = binary_search_iter(table, 0, w->num_voxels - 1, xyz); //iterative binary search

	if (vox_idx != -1) //"now intra"
	{
		if (tmp_loc == 1) { reject = false; } //was intra before
		else //was not intra before
		{
			//compute permeation probability p_21
			p_ex = (double)opt->kappa * sqrt(8 * opt->sim_dt / (3 * opt->D0)); //from Szafer 1995 and others, for any geoemetry
			p_12 = (double)p_ex * (1 - w->f1);
			p_21 = (double)p_ex * w->f1;
			if (curand_uniform_double(&states[index]) < p_21) { reject = false; tmp_loc = 1; }
			else { reject = true; tmp_loc = 0; }
		}
	}

	if (vox_idx == -1) //"now extra"
	{
		if (tmp_loc == 0) { reject = false; } //was extra before
		else //was not extra before
		{
			//compute permeation probability p_12
			p_ex = (double)opt->kappa * sqrt(8 * opt->sim_dt / (3 * opt->D0)); //from Szafer 1995 and others, for any geoemetry
			p_12 = (double)p_ex * (1 - w->f1);
			p_21 = (double)p_ex * w->f1;
			if (curand_uniform(&states[index]) < p_12) { reject = false; tmp_loc = 0; }
			else { reject = true; tmp_loc = 1; }
		}
	}


	if (reject) { tmp_x -= tmp_dx; tmp_y -= tmp_dy; tmp_z -= tmp_dz; vox_idx = old_vox_idx; }
	//restore voxel index if step is rejected.
}

__global__ //the global keyword tells compiler this is device code not host code
void engine(double* x, double* y, double* z, long long* loc, long long* table,
	curandState* states, options* opt, world* w, double* phase, double * gwf_x, double *gwf_y, double* gwf_z)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	//int stride = blockDim.x * gridDim.x;

	long entry, phase_entry;
	double tmp_x, tmp_y, tmp_z;
	double tmp_dx, tmp_dy, tmp_dz;
	double e_x, e_y, e_z; //keep track of hyperposition
	long long tmp_loc;
	long long vox_idx; // voxel index, needed by check_state

	//for (int c_p = index; c_p < (*opt).Npart; c_p += stride)
	long c_p = index;
	if (c_p < opt->Npart) //ensure we keep within bounds
	{
		tmp_x = x[c_p];
		tmp_y = y[c_p];
		tmp_z = z[c_p];
		tmp_dx = 0;
		tmp_dy = 0;
		tmp_dz = 0;
		tmp_loc = loc[c_p];

		//each thread needs to know the initial voxel id of its particle
		long long start_x_pos = floor(tmp_x / w->vox_size);
		long long start_y_pos = floor(tmp_y / w->vox_size);
		long long start_z_pos = floor(tmp_z / w->vox_size);
		long long start_xyz; //voxel identifier
		pair(start_x_pos, start_y_pos, start_z_pos, start_xyz); //get the identifier
		vox_idx = binary_search_iter(table, 0, w->num_voxels - 1, start_xyz); //iterative binary search

		//printf("vox_idx : %lld\n", vox_idx);
		
		e_x = 0; e_y = 0; e_z = 0;
		//adding a delay loop here
		
		for (int c_t = 0; c_t < (*opt).sim_Nt; c_t++)
		{
			move(tmp_x, tmp_y, tmp_z, tmp_dx,
				tmp_dy, tmp_dz, entry, states, index, opt);
			restrict_to_world(e_x, e_y, e_z, opt, w, tmp_x, tmp_y, tmp_z);
			check_state(w, opt, tmp_x, tmp_y, tmp_z, tmp_dx,
				tmp_dy, tmp_dz, tmp_loc, table, states, index, vox_idx);
			restrict_to_world(e_x, e_y, e_z, opt, w, tmp_x, tmp_y, tmp_z);
		}

		e_x = 0; e_y = 0; e_z = 0;

		//do signal-related calculations here
			  
			   //run the simulation
			for (int c_t = 0; c_t < (*opt).sim_Nt; c_t++)
				{
					move(tmp_x, tmp_y, tmp_z, tmp_dx,
					tmp_dy, tmp_dz, entry, states, index, opt);
					restrict_to_world(e_x, e_y, e_z, opt, w, tmp_x, tmp_y, tmp_z);
					check_state(w, opt, tmp_x, tmp_y, tmp_z, tmp_dx,
					tmp_dy, tmp_dz, tmp_loc, table, states, index, vox_idx);
					restrict_to_world(e_x, e_y, e_z, opt, w, tmp_x, tmp_y, tmp_z);

				for (int c_a = 0; c_a < opt->n_acq; c_a++)
					{
						 phase_entry = c_p * (*opt).n_acq + c_a;
			//printf("phase_entry : %f\n", tmp_y);
				phase[phase_entry] += opt->gamma*((tmp_x+e_x)*gwf_x[c_a*(*opt).sim_Nt+c_t] + (tmp_y+e_y)*gwf_y[c_a*(*opt).sim_Nt+c_t] + (tmp_z+e_z)*gwf_z[c_a*(*opt).sim_Nt+c_t])*opt->sim_dt;
				//if (c_p==0){printf("phase_entry : %f\n", gwf_x[c_a*(*opt).sim_Nt+c_t]);}
				//if (c_p==2){printf("phase_entry : %ld\n", phase_entry);}
				//if (c_p==2 && c_a==1){printf("phase_entry : %f\n", gwf_x[c_a*(*opt).sim_Nt+c_t]);}
					}
			}
	}
}



void set_options(options* opt, char* pos_fn, char* sig_fn, char* sub_fn, char* gwf_fn, char* sta_fn)
{
	//sets options loaded from file
	std::string opt_fn = "C:\\Users\\Arthur\\source\\repos\\PaSiD\\pasid_random_signal_opt.txt";
	std::string dummy;

	std::ifstream tf;
	tf.open(opt_fn, std::ios::in);
	tf >> dummy >> opt->Npart;
	tf >> dummy >> opt->sim_dt;
	tf >> dummy >> opt->samp_dt;
	tf >> dummy >> opt->D0;
	tf >> dummy >> opt->kappa;
	tf >> dummy >> sub_fn;
	tf >> dummy >> gwf_fn;
	tf >> dummy >> sig_fn;
	tf >> dummy >> opt->save_positions;
	tf >> dummy >> pos_fn;
	tf >> dummy >> opt->save_states;
	tf >> dummy >> sta_fn;
	tf.close();

	opt->delay = 500; //how many steps to take before starting to acquire signal
	opt->gamma = 2.675129e8;
	opt->n_dim = 3;
	opt->ds = (double)sqrt(2 * (*opt).n_dim * (*opt).D0 * (*opt).sim_dt); //step size
	
	std::cout << "Loaded options from: " << opt_fn << std::endl;
}


void save_trajectory(double* x, double* y, double* z, char* pos_fn, options* opt)
{
	std::cout << "Saving trajectory to: " << pos_fn << std::endl;
	FILE* tf;
	tf = fopen(pos_fn, "wb");
	fwrite(&(opt->Npart), sizeof(long long), 1, tf);
	fwrite(&(opt->T), sizeof(double), 1, tf);
	fwrite(&(opt->save_Nt), sizeof(long long), 1, tf);
	fwrite(x, sizeof(double), opt->N_save, tf); // opt->N_save
	fwrite(y, sizeof(double), opt->N_save, tf);
	fwrite(z, sizeof(double), opt->N_save, tf);
	fclose(tf);
	std::cout << "Done." << std::endl;
}


void save_state_history(long long* s, char* sta_fn, options* opt)
{
	//saves history of particle identities/compartment identities
	std::cout << "Saving state history to: " << sta_fn << std::endl;
	FILE* tf;
	tf = fopen(sta_fn, "wb");
	fwrite(&(opt->Npart), sizeof(long long), 1, tf);
	fwrite(&(opt->T), sizeof(double), 1, tf);
	fwrite(&(opt->save_Nt), sizeof(long long), 1, tf);
	fwrite(s, sizeof(long long), opt->Npart, tf); // opt->N_save
	fclose(tf);
	std::cout << "Done." << std::endl;
}


void save_signal(double* h_signal, char* sig_fn, options* opt)
{
	
	std::cout << "### Checking contents of signal..." << h_signal[0] << " " << h_signal[8] << std::endl;
	//saves signals to file
	std::cout << "Saving signals to: " << sig_fn << std::endl;
	FILE* tf;
	tf = fopen(sig_fn, "wb");
	fwrite(h_signal, sizeof(double), opt->n_acq, tf);
	fclose(tf);
	std::cout << "Done." << std::endl;
}

void get_num_voxels(world* w, char* sub_fn)
{
	//open substrate file and get number of voxels
	FILE* sf;
	sf = fopen(sub_fn, "rb");
	fread(&(w->num_voxels), sizeof(long long), 1, sf);
	fclose(sf);
}


void get_gwf_info(char* gwf_fn, options* opt)
{
	std::cout << "Getting gwf info from: " << gwf_fn << std::endl;
	//open gwf file and get number of b_values and number of time points in gwf
	FILE* sf;
	sf = fopen(gwf_fn, "rb");
	fread(&(opt->n_acq), sizeof(long long), 1, sf);
	fread(&(opt->n_gwf_points), sizeof(long long), 1, sf);
	fclose(sf);
	
	//total simulation time is number of points in gwf*sim_dt
	opt->sim_Nt = opt->n_gwf_points;
	//set this also in variable T
	opt->T = (opt->sim_Nt-1)*opt->sim_dt;
	
	opt->save_Nt = (long long)round(opt->T / opt->samp_dt);
	opt->N_save = (long long)opt->Npart * opt->save_Nt; //N time points x N particles
	opt->N_sim = (long long)opt->Npart * opt->sim_Nt;
	
	std::cout << "Done." << std::endl;
}


void load_substrate(long long* h_table, world* w, char* g_fn)
{
	//load substrate from file
	FILE* sf;
	sf = fopen(g_fn, "rb");
	fread(&(w->num_voxels), sizeof(long long), 1, sf);
	fread(&(w->vox_size), sizeof(double), 1, sf);
	fread(&(w->max_x), sizeof(double), 1, sf);
	fread(&(w->max_y), sizeof(double), 1, sf);
	fread(&(w->max_z), sizeof(double), 1, sf);
	fread(h_table, sizeof(long long), w->num_voxels, sf); //now expecting user to supply lookup table
	fread(&w->f1, sizeof(double), 1, sf);
	fclose(sf);

	w->x_length = 2 * w->max_x;
	w->y_length = 2 * w->max_y;
	w->z_length = 2 * w->max_z;

	std::cout << "Loaded substrate from: " << g_fn << std::endl;
}


void load_gwf(char* gwf_fn, options* opt, double* h_gwf_x, double* h_gwf_y, double* h_gwf_z)
{
	std::cout << "Loading gradient waveform..."<< std::endl;
	//load substrate from file
	FILE* sf;
	sf = fopen(gwf_fn, "rb");
	fread(&(opt->n_acq), sizeof(long long), 1, sf);
	fread(&(opt->n_gwf_points), sizeof(long long), 1, sf);
	fread(h_gwf_x, sizeof(double), opt->n_acq*opt->n_gwf_points, sf);
	fread(h_gwf_y, sizeof(double), opt->n_acq*opt->n_gwf_points, sf);
	fread(h_gwf_z, sizeof(double), opt->n_acq*opt->n_gwf_points, sf); 
	fclose(sf);

	std::cout << "$$$ Checking gwf: point 3 " << h_gwf_x[2] << " point 11: "<< h_gwf_x[10] <<std::endl;
	std::cout << "Loaded gradient waveform from: " << gwf_fn << std::endl;
}


void convert_phase_to_signal(double *h_phase, double *h_signal, options *opt)
{
	//convert phase to signal
	std::cout << "Converting phase to signal..." << std::endl;
	double sum_cos_phase;
	long long phase_entry;
	for (int ca = 0; ca<opt->n_acq; ca++)
	{
		sum_cos_phase = 0;
	for (long c_p = 0; c_p < opt->Npart; c_p++)
		{
			phase_entry = c_p * (*opt).n_acq + ca;
			sum_cos_phase += cos(h_phase[phase_entry]);
		}
		h_signal[ca] = sum_cos_phase/opt->Npart;
	}
	/*/
	for (int ca = 0; ca < opt->n_acq; ca++)
	{
		sum_cos_phase = 0;
		for (int cp = ca; cp < opt->n_acq*opt->Npart; cp+=opt->n_acq)
		{
			sum_cos_phase += cos(h_phase[cp]);
		}
		h_signal[ca] = sum_cos_phase/opt->Npart;
	 }/*/
	 std::cout << "Done." << std::endl;
}


void save_phase(double* h_phase, options *opt, char* sta_fn)
{
		//saves phase to file
		long long N = opt->Npart*opt->n_acq;
	std::cout << "Saving phase to: " << sta_fn << std::endl;
	FILE* tf;
	tf = fopen(sta_fn, "wb");
	fwrite(&N, sizeof(long long), 1, tf);
	fwrite(h_phase, sizeof(double), opt->n_acq*opt->Npart, tf);
	fclose(tf);
	std::cout << "Done." << std::endl;
}




__global__ void generate_initial_distribution(double* x, double* y, double* z, long long* loc, long long* table,
	curandState* states, options* opt, world* w)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //get thread idx
	double tmp_x, tmp_y, tmp_z, frac = 0.5;
	long long tmp_xll, tmp_yll, tmp_zll, tmp_xyz, tmp_vox_idx;
	long long idx;
	bool success_1 = false, success_2 = false, success = false;
	if (w->f1 == 0) { success_2 = true; frac = 1; if( index == 0) {printf("Detected f1 = 0.");};}
	int N1 = (int)(frac * opt->Npart);
	if (index < N1)
	{
		//places particles in initial positions all over substrate

		while (!success)
		{
			tmp_x = -w->max_x + 2 * curand_uniform_double(&states[index]) * w->max_x; //suggest initial position
			tmp_y = -w->max_y + 2 * curand_uniform_double(&states[index]) * w->max_y; //suggest initial position
			tmp_z = -w->max_z + 2 * curand_uniform_double(&states[index]) * w->max_z; //suggest initial position
			tmp_xll = (long long)floor(tmp_x / w->vox_size);
			tmp_yll = (long long)floor(tmp_y / w->vox_size);
			tmp_zll = (long long)floor(tmp_z / w->vox_size);
			pair(tmp_xll, tmp_yll, tmp_zll, tmp_xyz); //get pair
			tmp_vox_idx = binary_search_iter(table, 0, w->num_voxels - 1, tmp_xyz); //check what this position corresponds to

			//printf("%lld \n", tmp_vox_idx);
			
			if (tmp_vox_idx != -1 && !success_1)
			{
				idx = index;
				x[idx] = tmp_x;
				y[idx] = tmp_y;
				z[idx] = tmp_z;
				loc[idx] = 1;
				success_1 = true;
			}


			if (tmp_vox_idx == -1 && !success_2)
			{
				idx = (N1 + index);
				x[idx] = tmp_x;
				y[idx] = tmp_y;
				z[idx] = tmp_z;
				loc[idx] = 0;
				success_2 = true;
			}
			success = success_1 && success_2;
		}
	}
}


void generate_lookup_table(world* h_w, long long* h_world_x, long long* h_world_y, long long* h_world_z, long long* h_table)
{
	//build lookup table using the szudzik pairing algorithm
	std::cout << "Building voxel lookup table..." << std::endl;
	long long x, y, z, xy, xyz;
	for (int c = 0; c < h_w->num_voxels; c++)
	{
		x = h_world_x[c];
		y = h_world_y[c];
		z = h_world_z[c];

		x >= 0 ? x = 2 * x : x = -2 * x - 1;
		y >= 0 ? y = 2 * y : y = -2 * y - 1;
		z >= 0 ? z = 2 * z : z = -2 * z - 1;

		x >= y ? xy = x * x + x + y : xy = y * y + x;
		xy >= 0 ? xy = 2 * xy : xy = -2 * xy - 1;

		xy >= z ? xyz = xy * xy + xy + z : xyz = z * z + xy;

		h_table[c] = xyz;
	}
	std::cout << "Done." << std::endl;
}


int main(void)
{

	std::clock_t start;
	double duration;
	start = std::clock();

	cudaError error = cudaSuccess;

	int nDevices;
	cudaGetDeviceCount(&nDevices);
	printf("Number of devices: %d\n", nDevices);
	int activeDevice;
	cudaGetDevice(&activeDevice);
	printf("Active device index: %d\n", activeDevice);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, activeDevice);
	printf("Device name: %s\n", prop.name);


	//non-numerical options, host-only
	char pos_fn[200], sig_fn[200], sub_fn[200], gwf_fn[200], sta_fn[200]; //filenames for final positions, signal,  substrate, waveform, states

	//world structure for device and host
	world* h_w, * dev_w;
	h_w = (world*)malloc(sizeof(world));
	cudaMalloc(&dev_w, sizeof(world));


	//options structure for host
	options* opt, * dev_opt;
	opt = (options*)malloc(sizeof(options));
	set_options(opt, pos_fn, sig_fn, sub_fn, gwf_fn, sta_fn);
	
	//--GRADIENT WAVEFORM--
	//__________________________________________________________________
	double *h_gwf_x, *h_gwf_y, *h_gwf_z; //waveform in x y z
	double *dev_gwf_x, *dev_gwf_y, *dev_gwf_z;
	//get number of b-values and number of time points in gwf
	
	get_gwf_info(gwf_fn, opt); //need this to allocate memory for arrays above. Note: this function updates opt
	
	std::cout << "Number of acquisitions: " << opt->n_acq << std::endl;
	std::cout << "Number of time points per waveform: " << opt->n_gwf_points << std::endl;
	
	//allocate on host
	h_gwf_x = (double*)malloc(opt->n_acq*opt->n_gwf_points* sizeof(double));
	h_gwf_y = (double*)malloc(opt->n_acq*opt->n_gwf_points* sizeof(double));
	h_gwf_z = (double*)malloc(opt->n_acq*opt->n_gwf_points* sizeof(double));
	//allocate on device
	cudaMalloc(&dev_gwf_x, opt->n_acq*opt->n_gwf_points * sizeof(double));
	cudaMalloc(&dev_gwf_y, opt->n_acq*opt->n_gwf_points * sizeof(double));
	cudaMalloc(&dev_gwf_z, opt->n_acq*opt->n_gwf_points * sizeof(double));
	//load the waveform and b-values
	load_gwf(gwf_fn, opt, h_gwf_x, h_gwf_y, h_gwf_z);
	//copy to device
	cudaMemcpy(dev_gwf_x, h_gwf_x, opt->n_acq*opt->n_gwf_points * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gwf_y, h_gwf_y, opt->n_acq*opt->n_gwf_points * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gwf_z, h_gwf_z, opt->n_acq*opt->n_gwf_points * sizeof(double), cudaMemcpyHostToDevice);
	//__________________________________________________________________		
	cudaMalloc(&dev_opt, sizeof(options));
	cudaMemcpy(dev_opt, opt, sizeof(options), cudaMemcpyHostToDevice);

	//we will load the simulation world from file, no need to waste time implementing it in here
	//the first entry in the substrate file will be the number of voxels in the world
	//this is so we know how large the world arrays world_x, world_y, world_z and mask need to be

	get_num_voxels(h_w, sub_fn);
	std::cout << "Num vox intra: " << h_w->num_voxels << std::endl;
	//now we declare world arrays on device and host
	long long* h_table;
	long long* dev_table;

	//allocate on host
	h_table = (long long*)malloc(h_w->num_voxels * sizeof(long long));
	//allocate on device
	cudaMalloc(&dev_table, h_w->num_voxels * sizeof(long long));
	//load the substrate
	load_substrate(h_table, h_w, sub_fn);

	std::cout << "Num voxels: " << h_w->num_voxels << " Vox size: " << h_w->vox_size << " max_z: " << h_w->max_z << std::endl;

	//copy substrate data to GPU
	cudaMemcpy(dev_w, h_w, sizeof(world), cudaMemcpyHostToDevice);

	std::cout << " Example of table entry at ten: " << h_table[9] << std::endl;

	//copy the lookup table to the GPU
	error = cudaMemcpy(dev_table, h_table, h_w->num_voxels * sizeof(long long), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		std::cout << "FAILED TO ALLOCATE SUBSTRATE MEMORY ON GPU." << std::endl;
		//throw error;
	}


	//--SIGNAL--
	//declare signal and phase arrays
	double* h_signal, *h_phase; //for the host
	double * dev_phase; //for the device
	//allocate them on host
	h_signal = (double*)malloc(opt->n_acq * sizeof(double));
	h_phase = (double*)malloc(opt->n_acq*opt->Npart * sizeof(double));
	//initialise phase array, it's important that it contains only zeros
	for (int c = 0; c < opt->n_acq*opt->Npart; c++) {h_phase[c] = 0;}
	//allocate memory for arrays on device
	cudaMalloc(&dev_phase, opt->n_acq*opt->Npart * sizeof(double));
	//copy phase array to device
	cudaMemcpy(dev_phase, h_phase, opt->n_acq*opt->Npart * sizeof(double), cudaMemcpyHostToDevice);
	//__________________________________________________________________	

	//declare traj arrays and particle location (compartment id)
	long long* h_loc, * dev_loc;
	double* h_x, * h_y, * h_z; //for the host
	double* dev_x, * dev_y, * dev_z; //for the device
	//allocate them on host
	h_loc = (long long*)malloc(opt->Npart* sizeof(long long));
	h_x = (double*)malloc(opt->Npart * sizeof(double));
	h_y = (double*)malloc(opt->Npart * sizeof(double));
	h_z = (double*)malloc(opt->Npart * sizeof(double));


	//allocate memory for arrays on device
	cudaMalloc(&dev_loc, opt->Npart * sizeof(long long));
	cudaMalloc(&dev_x, opt->Npart * sizeof(double));
	cudaMalloc(&dev_y, opt->Npart * sizeof(double));
	cudaMalloc(&dev_z, opt->Npart * sizeof(double));

	//copy x,y,z  and id arrays to device
	cudaMemcpy(dev_loc, h_loc, opt->Npart * sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, h_x, opt->Npart * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, h_y, opt->Npart * sizeof(double), cudaMemcpyHostToDevice);
	error = cudaMemcpy(dev_z, h_z, opt->Npart * sizeof(double), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		std::cout << "FAILED TO ALLOCATE TRAJECTORY MEMORY ON GPU." << std::endl;
		//throw error;
	}
	else { std::cout << "Generating initial distribution..." << std::endl; }

	//define grid texture
	int blockSize;
	512 > opt->Npart ? blockSize = (int)opt->Npart : blockSize = 512;
	int numBlocks = (int)(opt->Npart + blockSize - 1) / blockSize; //make sure to round up in case N is not an integer multiple of blockSize

	//allcoate curandState for every CUDA thread on the host
	curandState* dev_states;
	cudaMalloc(&dev_states, blockSize * numBlocks * sizeof(curandState));
	//initialise RNG for all threads
	random_init << < numBlocks, blockSize >> > (dev_states);
	//generate initial particle distribution
	generate_initial_distribution << < numBlocks, blockSize >> > (dev_x, dev_y, dev_z, dev_loc, dev_table, dev_states, dev_opt, dev_w);
	//launch simulation engine
	std::cout << "Running simulation..." << std::endl;
	engine << < numBlocks, blockSize >> > (dev_x, dev_y, dev_z, dev_loc, dev_table, dev_states, dev_opt, dev_w, 
	dev_phase, dev_gwf_x, dev_gwf_y, dev_gwf_z);

	cudaDeviceSynchronize(); //Tell CPU to wait until kernel is done before accessing results. This is necessary because
							//cuda kernel launches do not block the calling CPU thread.

	std::cout << "Simulation complete. Downloading results..." << std::endl;

	//--DOWNLOAD RESULTS--
	//copy phase array back to host machine
	cudaMemcpy(h_phase, dev_phase, opt->n_acq*opt->Npart * sizeof(double), cudaMemcpyDeviceToHost);
	//copy final particle positions for diagnostics
	cudaMemcpy(h_x, dev_x, opt->Npart * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_y, dev_y, opt->Npart * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_z, dev_z, opt->Npart * sizeof(double), cudaMemcpyDeviceToHost);
	//__________________________________________________________________

	std::cout << "Done." << std::endl;
	
	convert_phase_to_signal(h_phase, h_signal, opt);
	
	save_phase(h_phase, opt, sta_fn);
	
	save_signal(h_signal, sig_fn, opt);
	
	//write results to binary files
	if (opt->save_positions) {save_trajectory(h_x, h_y, h_z, pos_fn, opt);};
	if (opt->save_states) {save_state_history(h_loc, sta_fn, opt);};

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "Elapsed time is: " << duration << " seconds." << std::endl;


	// Free memory on host
	free(h_signal);
	free(h_phase);
	free(h_gwf_x);
	free(h_gwf_y);
	free(h_gwf_z);
	free(h_x);
	free(h_y);
	free(h_z);
	free(h_w);
	free(opt);
	free(h_loc);
	free(h_table);
	//free memory on device
	//free memory on device
	cudaFree(dev_phase);
	cudaFree(dev_gwf_x);
	cudaFree(dev_gwf_y);
	cudaFree(dev_gwf_z);
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_z);
	cudaFree(dev_states);
	cudaFree(dev_loc);
	cudaFree(dev_w);
	cudaFree(dev_opt);
	cudaFree(dev_table);
	return 0;
}
