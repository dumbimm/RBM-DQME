#!/bin/sh


yourfolder=/home
commend_file=$yourfolder/NADO_torch_github/source
file1=$yourfolder/NADO_torch_github/examples/case2/T=\0.3/steady_state
file2=$yourfolder/NADO_torch_github/examples/case2/T=\0.3/td



# #1: Solving steady state with simulating rho
# cd $file1
# nohup python $commend_file/simulation_rho.py 1>simulate_nohup.out 2>&1 &
# wait
# cp $file1/rho_TNC $file2/para0
# wait

# #1-b: Solving steady state with random initial state
# cd $file1
# nohup python $commend_file/main_steady_state.py 1>td_steady_nohup.out 2>&1 &
# wait
# cp $file1/para_steady_state $file2/para0
# wait



# #2-a: Evolution with nonmc
# cd $file2
# cp input_nonmc input
# nohup python $commend_file/main_nonmc.py 1>td_nonmc_nohup.out 2>&1 &
# wait

# #2-b: Evolution with mc
# cd $file2
# cp input_mc input
# nohup python $commend_file/main_mc.py 1>td_mc_nohup.out 2>&1 &
# wait





## Run the following code:
# chmod +x job.sh
# nohup sh job.sh &