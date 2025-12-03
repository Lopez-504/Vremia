#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <curand_kernel.h>

#define IDX(x,y,L) ((y)*(L)+(x))

// ================= Random Number Generator =================
__global__ void init_rng(curandState *state, unsigned long long seed, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = L * L;
    if (idx < N) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// ================= Initialization =================
__global__ void init_spins_random(int8_t *spins, curandState *state, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = L * L;
    if (idx < N) {
        float r = curand_uniform(&state[idx]);		// Random number between 0 and 1
        spins[idx] = (r < 0.5f ? -1 : 1);		// if r< .5, spins[idx]= -1, else spins[idx]= 1 
    }
}

// Periodic Boundary Condition spin getter
__device__ inline int8_t get_spin(int8_t *spins, int x, int y, int L) {
    if (x < 0) x += L;
    if (x >= L) x -= L;
    if (y < 0) y += L;
    if (y >= L) y -= L;
    return spins[IDX(x, y, L)];
}

// ================= METROPOLIS UPDATE =================
__global__ void metropolis_update_color(int8_t *spins, curandState *state,
                                        float beta, int L, int color) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = L * L;
    if (idx >= N) return;

    int x = idx % L;
    int y = idx / L;

    if (((x + y) & 1) != color) return;

    int8_t s = spins[idx];

    int8_t nb = get_spin(spins, x+1, y,   L)
              + get_spin(spins, x-1, y,   L)
              + get_spin(spins, x,   y+1, L)
              + get_spin(spins, x,   y-1, L);

    int dE = 2 * s * nb;

    if (dE <= 0) {
        spins[idx] = -s;	//Step 4: if dE <=0, flip spin
    } else {
        float r = curand_uniform(&state[idx]);
        if (r < expf(-beta * dE)) spins[idx] = -s;	//step 5: if dE>0 flip spin with probability 1-p
    }
}

// ================= MAGNETIZATION =================
__global__ void compute_magnetization(int8_t *spins, int *out_sum, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = L * L;
    if (idx >= N) return;
    atomicAdd(out_sum, (int)spins[idx]);		
}


// ================= RUN ONE TEMPERATURE =================
void run_simulation_one(double T, int L,
                        int equil_sweeps, int sweeps_between_samples,
                        int n_samples, unsigned long long seed,
                        const std::string &out_csv,
                        const std::string &timing_csv,
                        int run_id)
{
    int N = L * L;
    size_t spins_bytes = N * sizeof(int8_t);
 
    // Alocate memory on GPU	
    int8_t *d_spins;
    cudaMalloc(&d_spins, spins_bytes);

    curandState *d_states;
    cudaMalloc(&d_states, N * sizeof(curandState));
	
    int *d_mag_sum;
    cudaMalloc(&d_mag_sum, sizeof(int));

    int threads = 256;					//try other configs 
    int blocks = (N + threads - 1) / threads;

    // Launch kernel: random number generator	
    init_rng<<<blocks, threads>>>(d_states, seed, L);
    cudaDeviceSynchronize();

    // Launch kernele: random initialize 	
    init_spins_random<<<blocks, threads>>>(d_spins, d_states, L);
    cudaDeviceSynchronize();

    float beta = 1.0f / T;

    // ---------- Timing accumulators ----------
    float total_sweep_time_ms = 0.0f;
    int timed_sweeps = 0;

    // ---------- CUDA event objects ----------
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // ---------- OPEN OUTPUT FILE ----------
    std::ofstream fout(out_csv, std::ios::app);
    if (!fout.is_open()) {
        printf("Error opening output file\n");
        return;
    }

    // Equilibration
    for (int sweep = 0; sweep < equil_sweeps; sweep++) {

        // start timer
        cudaEventRecord(ev_start);

        metropolis_update_color<<<blocks, threads>>>(d_spins, d_states, beta, L, 0);
        metropolis_update_color<<<blocks, threads>>>(d_spins, d_states, beta, L, 1);

        // stop timer
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);

        total_sweep_time_ms += ms;
        timed_sweeps++;
    }

    // Sampling 
    for (int s = 0; s < n_samples; s++) {

        for (int sweep = 0; sweep < sweeps_between_samples; sweep++) {

            cudaEventRecord(ev_start);

            metropolis_update_color<<<blocks, threads>>>(d_spins, d_states, beta, L, 0);
            metropolis_update_color<<<blocks, threads>>>(d_spins, d_states, beta, L, 1);

            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);

            float ms;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);

            total_sweep_time_ms += ms;
            timed_sweeps++;
        }

	// Initialize d_mag with 0s
        cudaDeviceSynchronize();
        cudaMemset(d_mag_sum, 0, sizeof(int));		
        
        // Launch kernel: magnetization 
        compute_magnetization<<<blocks, threads>>>(d_spins, d_mag_sum, L);
        cudaDeviceSynchronize();

	// device -> host
        int h_sum;
        cudaMemcpy(&h_sum, d_mag_sum, sizeof(int), cudaMemcpyDeviceToHost);

	// Compute <m> and write output file 
        double m = (double)h_sum / (double)N;
        fout << T << "," << run_id << "," << s << "," << m << "\n";
    }

    fout.close();

    // ---------- WRITE TIMING CSV ----------
    double avg_ms_per_sweep = total_sweep_time_ms / timed_sweeps;
    double ns_per_spin = (avg_ms_per_sweep * 1e6) / (2.0 * N);

    std::ofstream ft(timing_csv, std::ios::app);
    ft << T << "," << run_id << "," << avg_ms_per_sweep << "," << ns_per_spin << "\n";
    ft.close();

    // free
    cudaFree(d_spins);
    cudaFree(d_states);
    cudaFree(d_mag_sum);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}

// ================= MAIN =================
int main(int argc, char **argv)
{
    // Tried 32, 64 and 128, 256 
    int L = 256;
    if (argc > 1) L = atoi(argv[1]);		// try getting L from arguments

    // n_temps temperatures from T_min to T_max	
    int n_temps = 11;
    double T_min = 1.5;
    double T_max = 3.5;
    int runs_per_T = 8;
    int equil_sweeps = 5000;			// add more
    int sweeps_between_samples = 200;		// independence between states 
    int n_samples = 200;

    std::string out_csv = "magnetizations.csv";
    std::string timing_csv = "timings.csv";

    // Reset output files
    {
        std::ofstream f(out_csv, std::ios::trunc);
        f << "T,run_id,sample_idx,m\n";
    }
    {
        std::ofstream f(timing_csv, std::ios::trunc);
        f << "T,run_id,avg_sweep_ms,time_per_spin_ns\n";
    }

    // Fill temps
    std::vector<double> temps(n_temps);
    for (int i = 0; i < n_temps; i++)
        temps[i] = T_min + (T_max - T_min) * i / (n_temps - 1);

    int global_run_id = 0;

    // Loop over temperatures in temps
    for (double T : temps) {
        printf("Starting T = %.4f\n", T);
        for (int r = 0; r < runs_per_T; r++) {
            unsigned long long seed =
                123456789ULL + (unsigned long long)global_run_id * 7919ULL;

            run_simulation_one(
                T, L, equil_sweeps, sweeps_between_samples, n_samples,
                seed, out_csv, timing_csv, global_run_id
            );

            global_run_id++;
        }
    }

    printf("All done.\n");
    return 0;
}

