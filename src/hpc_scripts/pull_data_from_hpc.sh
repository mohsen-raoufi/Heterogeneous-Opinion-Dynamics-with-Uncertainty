# Copy this git project to MKP cluster
​
# Get the path to the top level of this git repository
# local_path=$(git rev-parse --show-toplevel)
local_path=$(pwd -P)
​
# Pull the simulation files from the remote folder
# rsync -avz m.raoufi.s@gateway.hpc.tu-berlin.de:/home/users/m/m.raoufi.s/colab/collective-decison-making-with-direl/results/2022-12-16-15-12-58_test_grid_search_Bayes/envstd_0.5556444444444445_mnmax_0.5556444444444445_mnmin_0.5556444444444445_np1_0.8421052631578947_ow_0.0_a_n_100_run_18 "$local_path"/
rsync -az --info=progress2 m.raoufi.s@gateway.hpc.tu-berlin.de:/home/users/m/m.raoufi.s/colab/collective-decison-making-with-direl/results/N100_2023-08-18-17-16-12_network_search_Naive_centralized_random_fixed_mdeg "$local_path"/results/
