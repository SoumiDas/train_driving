1. Please install the dependencies using environment.yml via anaconda3:
    <b> conda env create -f environment.yml </b>
2. Once those are installed, activate the environment:
    <b> conda activate CAL </b>
3. In order to run the training, run
    <b> python training_cal.py </b>
4. I have kept only 3 episodes here as data: 2 for training, 1 for validation.
5. This is going to run for 1 epoch only (the parameter is present in the last line of training_cal.py as the first argument of fit() )
6. Intel Xeon 8 core, 32GB RAM, NVIDIA GTX 1080Ti 11GB takes around 33seconds for this particular setup.
