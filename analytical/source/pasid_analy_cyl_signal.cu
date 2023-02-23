#include <iostream>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <fstream>
#include <stdio.h>
#include <random>
//This program simulates restricted diffusion with exchange in a substrate of regular cylinders defined analytically
//Cylinders may have any packing and any size distribution as long as they are parallel
//Can also save trajectories
//Always saves signal
//Reads one gradient waveform from binary file.
//Computes signal in 2D for now (x y)

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
	long long n_b_values; //# b-values,
	long long n_gwf_points; // # time points in each waveform
	double max_b_value;
	double gamma; //gyromagnetic ratio
	long long delay; //number of time steps to take before acquiring signals
	bool save_positions; //save final positions or not; useful for debugging
};


//world data
struct world
{
	long long num_cells, num_voxels;
	double max_x, max_y, max_z, x_length, y_length, z_length, f1, vox_size;
};

__global__ void random_init(curandState* states)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock64()+index, index, 0, &states[index]);
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


__device__ void pair(long long x, long long y, long long& xy)
{
	//device function for pairing particle coordinates
	x >= 0 ? x = 2 * x : x = -2 * x - 1;
	y >= 0 ? y = 2 * y : y = -2 * y - 1;

	x >= y ? xy = x * x + x + y : xy = y * y + x;
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



__device__ long long is_particle_in_any_cell(double tmp_x, double tmp_y, world* w, double* centre_x, double* centre_y, double* radii, long long* table, long long* cell_idx)
{
	long long x_pos = floor(tmp_x / w->vox_size);
	long long y_pos = floor(tmp_y / w->vox_size);
	long long xy; //voxel identifier
	long long which_cell, which_voxel;
	long long inside = 0; //zero means outside all cells
	double distance, r, cx, cy;
	
	pair(x_pos, y_pos, xy); //get the identifier


	which_voxel = binary_search_iter(table, 0, w->num_voxels - 1, xy); //iterative binary search	
	
	//printf("--------\n");
	//printf("which_voxel is: %lld\n", which_voxel);
	//printf("xy is: %lld\n", xy);
	
	which_cell = cell_idx[which_voxel]-1; //index of cell containing the voxel containing the particle
	//note the minus 1 to take into account that MATLAB numbering starts at 1. Need to fix this later
				
    //printf("which_cell is: %lld\n", which_cell);
    
    r = radii[which_cell];
    cx = centre_x[which_cell];
    cy = centre_y[which_cell];

	//printf("cell_idx is: %lld\n", which_cell);
	
	if (which_cell >= 0) //-1 means voxel is not in any cell
	{
		distance = (tmp_x-cx)*(tmp_x-cx) + (tmp_y-cy)*(tmp_y-cy);
		if ( distance <= (r*r)  ) //means particle is in this cell
		{
			inside = 1;
		}
	}
	/*/
	printf("Is inside? : %lld\n", inside);
	printf("tmp_x : %f\n", tmp_x*1e6);
	printf("tmp_y : %f\n", tmp_y*1e6);
	printf("Dist : %f\n", distance*1e12);
		printf("cx : %f\n", cx*1e6);
		printf("cy : %f\n", cy*1e6);
		printf("r : %f\n", r*1e6);
	/*/
	
	return inside;
}

__device__ void check_state(world* w, options* opt, double& tmp_x, double& tmp_y, double& tmp_z, double& tmp_dx,
	double& tmp_dy, double& tmp_dz, long long& tmp_loc, double* centre_x, double* centre_y, double *radii, long long* table, long long * cell_idx,  curandState* states, int index)
{
	bool reject = false; //reject move or not
	double p_ex, p_12, p_21;
	long long inside;

	inside = is_particle_in_any_cell(tmp_x, tmp_y, w, centre_x, centre_y, radii, table, cell_idx); //determine whether particle is in any cell

	if (inside == 1) //"now intra"
	{
		if (tmp_loc == 1) { reject = false; } //was intra before
		else //was not intra before
		{
			//compute permeation probability p_21
			p_ex = (double)opt->kappa * sqrt(8 * opt->sim_dt / (3 * opt->D0)); //from Szafer 1995 and others, for any geoemetry
			p_21 = (double)p_ex * w->f1;
			if (curand_uniform_double(&states[index]) < p_21) { reject = false; tmp_loc = 1; }
			else { reject = true; tmp_loc = 0; }
		}
	}

	if (inside == 0) //"now extra"
	{
		if (tmp_loc == 0) { reject = false; } //was extra before
		else //was not extra before
		{
			//compute permeation probability p_12
			p_ex = (double)opt->kappa * sqrt(8 * opt->sim_dt / (3 * opt->D0)); //from Szafer 1995 and others, for any geoemetry
			p_12 = (double)p_ex * (1 - w->f1);
			if (curand_uniform_double(&states[index]) < p_12) { reject = false; tmp_loc = 0; }
			else { reject = true; tmp_loc = 1; }
		}
	}


	if (reject) { tmp_x -= tmp_dx; tmp_y -= tmp_dy; tmp_z -= tmp_dz; }

}

__global__ //the global keyword tells compiler this is device code not host code
void engine(double* x, double* y, double* z, long long* loc, double* centre_x, double* centre_y, double* radii, long long* table, long long* cell_idx,
	curandState* states, options* opt, world* w, double* b_values, double* phase, double * gwf_x, double *gwf_y, double* gwf_z)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	//int stride = blockDim.x * gridDim.x;

	long entry, phase_entry;
	long save;
	long save_c_t;
	double tmp_x, tmp_y, tmp_z;
	double tmp_dx, tmp_dy, tmp_dz;
	double e_x, e_y, e_z; //keep track of hyperposition
	long long tmp_loc;

	double gf; //factor by which to scale gwf to change b-value

	//for (int c_p = index; c_p < (*opt).Npart; c_p += stride)
	long c_p = index;
	if (c_p < opt->Npart) //ensure we keep within bounds
	{
		tmp_x = x[(long)(c_p * (*opt).save_Nt)];
		tmp_y = y[(long)(c_p * (*opt).save_Nt)];
		tmp_z = z[(long)(c_p * (*opt).save_Nt)];
		tmp_dx = 0;
		tmp_dy = 0;
		tmp_dz = 0;
		tmp_loc = loc[(long)(c_p * (*opt).save_Nt)];

		e_x = 0; e_y = 0; e_z = 0;
		//adding a delay loop here
		
		
		for (int c_t = 0; c_t < (*opt).delay; c_t++)
		{
			move(tmp_x, tmp_y, tmp_z, tmp_dx,
				tmp_dy, tmp_dz, entry, states, index, opt);
			restrict_to_world(e_x, e_y, e_z, opt, w, tmp_x, tmp_y, tmp_z);
			check_state(w, opt, tmp_x, tmp_y, tmp_z, tmp_dx,
				tmp_dy, tmp_dz, tmp_loc, centre_x, centre_y, radii, table, cell_idx, states, index);
			restrict_to_world(e_x, e_y, e_z, opt, w, tmp_x, tmp_y, tmp_z);
		}

		e_x = 0; e_y = 0; e_z = 0;

		save_c_t = -1;
		save = 0;

		for (int c_t = 0; c_t < (*opt).sim_Nt; c_t++)
		{

			move(tmp_x, tmp_y, tmp_z, tmp_dx,
				tmp_dy, tmp_dz, entry, states, index, opt);
			restrict_to_world(e_x, e_y, e_z, opt, w, tmp_x, tmp_y, tmp_z);
			check_state(w, opt, tmp_x, tmp_y, tmp_z, tmp_dx,
				tmp_dy, tmp_dz, tmp_loc, centre_x, centre_y, radii, table, cell_idx, states, index);
			restrict_to_world(e_x, e_y, e_z, opt, w, tmp_x, tmp_y, tmp_z);

		    //do signal-related calculations here
			for (int c_s = 0; c_s < opt->n_b_values; c_s++)
			{
				gf = sqrt(b_values[c_s]/opt->max_b_value);
				phase_entry = c_p * (*opt).n_b_values + c_s;
				phase[phase_entry] += opt->gamma*((tmp_x+e_x)*gwf_x[c_t]*gf + (tmp_y+e_y)*gwf_y[c_t]*gf + (tmp_z+e_z)*gwf_z[c_t]*gf)*opt->sim_dt;
				//if (c_p==2){printf("phase_entry : %ld\n", phase_entry);}
			}
			
			save++;
			
			if (save == (long)(opt->samp_dt / opt->sim_dt) )
			{
				save_c_t++;
				entry = c_p * (*opt).save_Nt + save_c_t;
				x[entry] = tmp_x + e_x;
				y[entry] = tmp_y + e_y;
				z[entry] = tmp_z + e_z;
				loc[entry] = tmp_loc;
				save = 0;
				//if (true) {printf("tmp_x : %f\n", x[entry]*1e6);};
			}
		}
	}
}


void set_options(options* opt, char* pos_fn, char* sig_fn, char* sub_fn, char* gwf_fn, char* sta_fn)
{
	//sets options loaded from file
	std::string opt_fn = "C:\\Users\\Arthur\\source\\repos\\PaSiD\\pasid_analy_cyl_sig_opt.txt";
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
	fwrite(s, sizeof(long long), opt->N_save, tf); // opt->N_save
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
	fwrite(h_signal, sizeof(double), opt->n_b_values, tf);
	fclose(tf);
	std::cout << "Done." << std::endl;
}



void get_num_cells(world* w, char* sub_fn)
{
	//open substrate file and get number of cells and number of voxels
	FILE* sf;
	sf = fopen(sub_fn, "rb");
	fread(&(w->num_cells), sizeof(long long), 1, sf);
	fread(&(w->num_voxels), sizeof(long long), 1, sf);
	fclose(sf);
}


void get_gwf_info(char* gwf_fn, options* opt)
{
	std::cout << "Getting gwf info from: " << gwf_fn << std::endl;
	//open gwf file and get number of b_values and number of time points in gwf
	FILE* sf;
	sf = fopen(gwf_fn, "rb");
	fread(&(opt->n_b_values), sizeof(long long), 1, sf);
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

void load_substrate(double* h_centre_x, double* h_centre_y, double* h_radii, long long* h_table, long long *h_cell_idx, world* w, char* g_fn)
{
	//load substrate from file
	FILE* sf;
	sf = fopen(g_fn, "rb");
	fread(&(w->num_cells), sizeof(long long), 1, sf);
	fread(&(w->num_voxels), sizeof(long long), 1, sf);
	fread(&(w->max_x), sizeof(double), 1, sf);
	fread(&(w->max_y), sizeof(double), 1, sf);
	fread(&(w->max_z), sizeof(double), 1, sf);
	fread(&(w->vox_size), sizeof(double), 1, sf);
	fread(&w->f1, sizeof(double), 1, sf);
	fread(h_centre_x, sizeof(double), w->num_cells, sf); 
	fread(h_centre_y, sizeof(double), w->num_cells, sf); 
	fread(h_radii, sizeof(double), w->num_cells, sf); 
	fread(h_table, sizeof(long long), w->num_voxels, sf); 
	fread(h_cell_idx, sizeof(long long), w->num_voxels, sf); 
	fclose(sf);

	w->x_length = 2 * w->max_x;
	w->y_length = 2 * w->max_y;
	w->z_length = 2 * w->max_z;

	std::cout << "Loaded substrate from: " << g_fn << std::endl;
}


void load_gwf(char* gwf_fn, options* opt, double* h_b_values, double* h_gwf_x, double* h_gwf_y, double* h_gwf_z)
{
	std::cout << "Loading gradient waveform..."<< std::endl;
	//load substrate from file
	FILE* sf;
	sf = fopen(gwf_fn, "rb");
	fread(&(opt->n_b_values), sizeof(long long), 1, sf);
	fread(&(opt->n_gwf_points), sizeof(long long), 1, sf);
	fread(h_b_values, sizeof(double), opt->n_b_values, sf);
	fread(h_gwf_x, sizeof(double), opt->n_gwf_points, sf);
	fread(h_gwf_y, sizeof(double), opt->n_gwf_points, sf);
	fread(h_gwf_z, sizeof(double), opt->n_gwf_points, sf); 
	fclose(sf);

	//put max b value in opt
	double max_b_value = 0;
	for (int c = 0; c < opt->n_b_values; c++) {if (h_b_values[c] > max_b_value) max_b_value = h_b_values[c];}
	opt->max_b_value = max_b_value;
	std::cout << "Loaded gradient waveform from: " << gwf_fn << std::endl;
}


void convert_phase_to_signal(double *h_phase, double *h_signal, options *opt)
{
	//convert phase to signal
	std::cout << "Converting phase to signal..." << std::endl;
	double sum_cos_phase;
	for (int cb = 0; cb < opt->n_b_values; cb++)
	{
		sum_cos_phase = 0;
		for (int cp = cb; cp < opt->n_b_values*opt->Npart; cp+=opt->n_b_values)
		{
			sum_cos_phase += cos(h_phase[cp]);
		}
		h_signal[cb] = sum_cos_phase/opt->Npart;
	 }
	 std::cout << "Done." << std::endl;
}

void save_phase(double* h_phase, options *opt, char* sta_fn)
{
		//saves phase to file
		long long N = opt->Npart*opt->n_b_values;
	std::cout << "Saving phase to: " << sta_fn << std::endl;
	FILE* tf;
	tf = fopen(sta_fn, "wb");
	fwrite(&N, sizeof(long long), 1, tf);
	fwrite(h_phase, sizeof(double), opt->n_b_values*opt->Npart, tf);
	fclose(tf);
	std::cout << "Done." << std::endl;
}


__global__ void generate_initial_distribution(double* x, double* y, double* z, long long* loc, double* centre_x,
	double* centre_y, double* radii, long long* table, long long* cell_idx, curandState* states, options* opt, world* w)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; //get thread idx
	double tmp_x, tmp_y, tmp_z, frac = 0.5; //frac defines intra-extra split of initial populations
	long long idx, inside;
	bool success_1 = false, success_2 = false, success = false;
	if (w->f1 == 0) { success_2 = true; frac = 1; }
	int N1 = (int)(frac * opt->Npart);
	if (index < N1)
	{
		//places particles in initial positions all over substrate

		while (!success)
		{
			tmp_x = -w->max_x + 2 * curand_uniform_double(&states[index]) * w->max_x; //suggest initial position
			tmp_y = -w->max_y + 2 * curand_uniform_double(&states[index]) * w->max_y; //suggest initial position
			tmp_z = -w->max_z + 2 * curand_uniform_double(&states[index]) * w->max_z; //suggest initial position
			
			inside = is_particle_in_any_cell(tmp_x, tmp_y, w, centre_x, centre_y, radii, table, cell_idx); //determine whether particle is in any cell
			
			//if (index == 1 && inside == 1) {printf("cell_idx is: %lld\n", inside); printf("x_pos: %f\n", tmp_x*1e6);  printf("y_pos: %f\n", tmp_y*1e6);  printf("z_pos: %f\n", tmp_z*1e6);};
			//printf("cell_idx is: %lld\n", inside);
			
			if (inside == 1 && !success_1)
			{
				idx = index * opt->save_Nt;
				x[idx] = tmp_x;
				y[idx] = tmp_y;
				z[idx] = tmp_z;
				loc[idx] = 1;
				success_1 = true;
			}


			if (inside == 0 && !success_2)
			{
				idx = (N1 + index) * opt->save_Nt;
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
	double *h_b_values, *dev_b_values;
	double *h_gwf_x, *h_gwf_y, *h_gwf_z; //waveform in x y z
	double *dev_gwf_x, *dev_gwf_y, *dev_gwf_z;
	//get number of b-values and number of time points in gwf
	
	get_gwf_info(gwf_fn, opt); //need this to allocate memory for arrays above. Note: this function updates opt
	
	std::cout << "Number of b-values: " << opt->n_b_values << std::endl;
	std::cout << "Number of time points: " << opt->n_gwf_points << std::endl;
	
	//allocate on host
	h_b_values = (double*)malloc(opt->n_b_values* sizeof(double));
	h_gwf_x = (double*)malloc(opt->n_gwf_points* sizeof(double));
	h_gwf_y = (double*)malloc(opt->n_gwf_points* sizeof(double));
	h_gwf_z = (double*)malloc(opt->n_gwf_points* sizeof(double));
	//allocate on device
	cudaMalloc(&dev_b_values, opt->n_b_values * sizeof(double));
	cudaMalloc(&dev_gwf_x, opt->n_gwf_points * sizeof(double));
	cudaMalloc(&dev_gwf_y, opt->n_gwf_points * sizeof(double));
	cudaMalloc(&dev_gwf_z, opt->n_gwf_points * sizeof(double));
	//load the waveform and b-values
	load_gwf(gwf_fn, opt, h_b_values, h_gwf_x, h_gwf_y, h_gwf_z);
	//copy to device
	cudaMemcpy(dev_b_values, h_b_values, opt->n_b_values * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gwf_x, h_gwf_x, opt->n_gwf_points * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gwf_y, h_gwf_y, opt->n_gwf_points * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gwf_z, h_gwf_z, opt->n_gwf_points * sizeof(double), cudaMemcpyHostToDevice);
	//__________________________________________________________________		
	cudaMalloc(&dev_opt, sizeof(options));
	cudaMemcpy(dev_opt, opt, sizeof(options), cudaMemcpyHostToDevice);

	//we will load the simulation world from file, no need to waste time implementing it in here
	//the first entry in the substrate file will be the number of cells in the world
	//this is so we know how large the world arrays centre_x, centre_y, radii need to be

	get_num_cells(h_w, sub_fn);
	std::cout << "Num cells: " << h_w->num_cells << "Num voxels: " << h_w->num_voxels << std::endl;
	//now we declare world arrays on device and host
	double* h_centre_x, *h_centre_y, *h_radii;
	double* dev_centre_x, *dev_centre_y, *dev_radii;
	long long *h_table, *h_cell_idx;
	long long *dev_table, *dev_cell_idx;
	
	
	//allocate on host
	h_centre_x = (double*)malloc(h_w->num_cells * sizeof(double));
	h_centre_y = (double*)malloc(h_w->num_cells * sizeof(double));
	h_radii = (double*)malloc(h_w->num_cells * sizeof(double));
	h_table = (long long*)malloc(h_w->num_voxels * sizeof(long long));
	h_cell_idx = (long long*)malloc(h_w->num_voxels * sizeof(long long));
	
	//allocate on device
	cudaMalloc(&dev_centre_x, h_w->num_cells * sizeof(double));
	cudaMalloc(&dev_centre_y, h_w->num_cells * sizeof(double));
	cudaMalloc(&dev_radii, h_w->num_cells * sizeof(double));
	cudaMalloc(&dev_table, h_w->num_voxels * sizeof(long long));
	cudaMalloc(&dev_cell_idx, h_w->num_voxels * sizeof(long long));
	
	//load the substrate
	load_substrate(h_centre_x, h_centre_y, h_radii, h_table, h_cell_idx, h_w, sub_fn);

	std::cout << "Num cells: " << h_w->num_cells << " Diameter[0]: " << h_radii[0]*2 << 
	" max_z: " << h_w->max_z << " Num voxels: " << h_w->num_voxels  << 
	" Voxel size : " << h_w->vox_size << " Table(23) : " << h_table[22] << std::endl;

	//copy substrate data to GPU
	cudaMemcpy(dev_w, h_w, sizeof(world), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_centre_x, h_centre_x, h_w->num_cells * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_centre_y, h_centre_y, h_w->num_cells * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_radii, h_radii, h_w->num_cells * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_table, h_table, h_w->num_voxels * sizeof(long long), cudaMemcpyHostToDevice);
	error = cudaMemcpy(dev_cell_idx, h_cell_idx, h_w->num_voxels * sizeof(long long), cudaMemcpyHostToDevice);
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
	h_signal = (double*)malloc(opt->n_b_values * sizeof(double));
	h_phase = (double*)malloc(opt->n_b_values*opt->Npart * sizeof(double));
	//initialise phase array, it's important that it contains only zeros
	for (int c = 0; c < opt->n_b_values*opt->Npart; c++) {h_phase[c] = 0;}
	//allocate memory for arrays on device
	cudaMalloc(&dev_phase, opt->n_b_values*opt->Npart * sizeof(double));
	//copy phase array to device
	cudaMemcpy(dev_phase, h_phase, opt->n_b_values*opt->Npart * sizeof(double), cudaMemcpyHostToDevice);
	//__________________________________________________________________	


	//declare traj arrays and particle location (compartment id)
	long long* h_loc, * dev_loc;
	double* h_x, * h_y, * h_z; //for the host
	double* dev_x, * dev_y, * dev_z; //for the device
	//allocate them on host
	h_loc = (long long*)malloc(opt->N_save * sizeof(long long));
	h_x = (double*)malloc(opt->N_save * sizeof(double));
	h_y = (double*)malloc(opt->N_save * sizeof(double));
	h_z = (double*)malloc(opt->N_save * sizeof(double));


	//allocate memory for arrays on device
	cudaMalloc(&dev_loc, opt->N_save * sizeof(long long));
	cudaMalloc(&dev_x, opt->N_save * sizeof(double));
	cudaMalloc(&dev_y, opt->N_save * sizeof(double));
	cudaMalloc(&dev_z, opt->N_save * sizeof(double));

	//copy x,y,z  and id arrays to device
	cudaMemcpy(dev_loc, h_loc, opt->N_save * sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, h_x, opt->N_save * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, h_y, opt->N_save * sizeof(double), cudaMemcpyHostToDevice);
	error = cudaMemcpy(dev_z, h_z, opt->N_save * sizeof(double), cudaMemcpyHostToDevice);
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
	
	
	generate_initial_distribution << < numBlocks, blockSize >> > (dev_x, dev_y, dev_z, dev_loc, dev_centre_x, dev_centre_y, dev_radii, dev_table, dev_cell_idx, dev_states, dev_opt, dev_w);

	std::cout << "Running simulation..." << std::endl;
	//launch simulation engine
	engine << < numBlocks, blockSize >> > (dev_x, dev_y, dev_z, dev_loc, dev_centre_x, dev_centre_y, dev_radii, dev_table, dev_cell_idx, dev_states, dev_opt, dev_w,
	dev_b_values, dev_phase, dev_gwf_x, dev_gwf_y, dev_gwf_z);

	cudaDeviceSynchronize(); //Tell CPU to wait until kernel is done before accessing results. This is necessary because
							//cuda kernel launches do not block the calling CPU thread.

	std::cout << "Simulation complete. Downloading results..." << std::endl;

	//copy simulated trajectories back to host machine
	cudaMemcpy(h_x, dev_x, opt->N_save * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_y, dev_y, opt->N_save * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_z, dev_z, opt->N_save * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_loc, dev_loc, opt->N_save * sizeof(long long), cudaMemcpyDeviceToHost);
	
	//copy phase array back to host machine
	cudaMemcpy(h_phase, dev_phase, opt->n_b_values*opt->Npart * sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << "Done." << std::endl;
	
	convert_phase_to_signal(h_phase, h_signal, opt);
	
	//save_phase(h_phase, opt, sta_fn);
	
	save_signal(h_signal, sig_fn, opt);

	//write results to binary files
	if (opt->save_positions) {save_trajectory(h_x, h_y, h_z, pos_fn, opt);};
	if (opt->save_states) {save_state_history(h_loc, sta_fn, opt);};
	
	
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "Elapsed time is: " << duration << " seconds." << std::endl;	


	//--CLEAN UP--
	// Free memory on host
	free(h_signal);
	free(h_phase);
	free(h_b_values);
	free(h_gwf_x);
	free(h_gwf_y);
	free(h_gwf_z);
	free(h_x);
	free(h_y);
	free(h_z);
	free(h_w);
	free(opt);
	free(h_loc);
	free(h_centre_x);
	free(h_centre_y);
	free(h_radii);
	free(h_table);
	free(h_cell_idx);
	//free memory on device
	//free memory on device
	cudaFree(dev_phase);
	cudaFree(dev_b_values);
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
	cudaFree(dev_centre_x);
	cudaFree(dev_centre_y);
	cudaFree(dev_radii);
	cudaFree(dev_table);
	cudaFree(dev_cell_idx);
	return 0;
}
