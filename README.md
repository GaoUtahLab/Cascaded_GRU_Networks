# Cascaded GRU Networks

## General info:

Three folders ```meta_material_demo```, ```graphene_demo```, and ```landau_polariton``` are included in this directory. Each of them contains the demo code for the applications in the paper (metamaterial, graphene, Landau polariton experiment). The MATLAB codes for calculating Landau polariton dispersions using transfer matrix are included in the folder ```Matlab Code```. Please refer to the README file in the folder for the instructions of running the MATLAB code. All original files of presented data are included in the zip file ```All_original_data.zip```.


### Environment used:

```bash
1. python (3.7)
1. pytorch  (1.2)
2. numpy (1.19.2)
3. scipy (1.6.1)
4. matplotlib (3.3.4)
5. scikit-learn (0.24.1)
```
One way to set up a virtual environment using conda is:

```bash
% conda create -n cascaded_gru numpy=1.19.2 scipy=1.6.1 matplotlib=3.3.4 python=3.7 scikit-learn=0.24.1
% conda activate cascaded_gru

Pytorch is installed on OSX in demo. Other platforms can be installed following instructions: https://pytorch.org/get-started/previous-versions/
% conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```

### Folder info:

1. Under ```meta_material_demo``` folder, it contains two subfolders ```input_sequence``` and ```fully_trained_model```.
Under the folder ```fully_trained_model```, it contains all the trained blocks for cascaded GRU networks:

```bash
1. 'GRU_model_same_seq_200_400_final.pkl'
2. 'GRU_model_same_seq_400_600_final.pkl'
3. 'GRU_model_same_seq_600_800_final.pkl'
4. 'GRU_model_same_seq_800_1200_final.pkl'
5. 'GRU_model_same_seq_1200_1600_final.pkl'
```

with the dataset 

```bash
'Ex_data_average_final_set.mat' 
```

and the dataloader 

```bash
'dataset_full_test.py'
```

When running the file ```metamaterial_demo.py``` under the ```fully_trained_model``` folder, the program will generate the predicted waveform data(```test_output_metamaterial.mat```) along with a sample predicted waveform plot (```sample_raw_prediction_waveform.pdf```). The dataset(```Ex_data_average_final_set.mat```) contains the time series data of the Electric Field (Ex) transmission of dielectric metamaterials. The data is collected through FDTD simulation.

Under ```input_sequence``` folder, it contains a testing script (```input_sequence_test.py```) that will generate the test loss with different input sequence lengths. It will also generate the predicted waveform data(```input_seq_xxx_result.mat```) along with a sample plot (```input_seq_xxx_result.pdf```).


2. Under the ```graphene_demo``` folder, similar to the metamaterial folder, it also contains the trained models, the test dataset (```dataset_app3_new_5_diff.mat```), and the demo python script(```graphene_demo.py```) along with the dataloader file(```dataset_full_p3_test_final.py```).
The dataset(```dataset_app3_new_5_diff.mat```) contains the time series data of the Electric Field (Ex) transmission of graphene plasmonic structures. The data is collected through FDTD simulation.

The blocks for cascaded GRU network models are listed as:

```bash
1. 'GRU_model_same_seq_45_normal_new_m5.pkl'.
2. 'GRU_model_same_seq_75_normal_new_m5.pkl'.
3. 'GRU_model_same_seq_150_normal_new_m5.pkl'.
4. 'GRU_model_same_seq_300_normal_new_m5.pkl'.
```

For models we used in the main content of the figure, the two models ```GRU_model_same_seq_150_normal_new_m5.pkl``` and ```GRU_model_same_seq_300_normal_new_m5.pkl``` are used to predict the 450 sequence-long data with input data length of 150.


When running ```graphene_demo.py```, the program will load the testing set and the trained model to predict the output waveform. It will generate one example predicted figure(```sample_raw_prediction_waveform.pdf```)
along with a .mat (```test_output_graphene.mat```) file that contains all the predict results and target results.


The other script ```graphene_input_sequence_test.py``` is used as the demonstration for the input sequence length sweep results. After running the file, it will print out the test loss based on different input sequence lengths (40 and 75). It will also generate one example figure(```input_seq_xx_result.pdf```) along with a .mat (```input_seq_xx_result.mat```) file that contains all the predict results and target results.


3. Under the 'landau_polariton_demo' folder, it contains six subfolders which includes the experiments we performed. They are:

```bash
1. 'input_sequence' folder contains the script for the test loss script for different input sequence lengths sweeping.
2. 'noise' folder contains the test loss script for the data with different A (from 0.01 to 0.2).
3. 'sampling' folder contains the test loss script for data with different sampling rate (from 0.5 to 20).
4. 'shuffle' folder contains the training script with shuffled input data.
5. 'visualization' folder contains the visualization code for the intermediate hidden states.
6. 'fully_trained_model' folder contains the trained models for the time-domain waveform and associated resonance frequencies, the test dataset('dataset_test.mat'), and the demo python script('landau_polariton_demo.py') along with the dataloader file('dataset_full2_p3_all.py').
```

Under each file, it contains the test dataset(```dataset_wave.mat```) along with the dataloader file(```dataset_full_waveform.py```).
The dataset(```dataset_wave.mat```) contains the experiment data collected through terahertz time-domain spectroscopy (THz-TDS) measurement.


Under ```fully_trained_model``` folder, it contains the model we used in the main content of the figure.
The blocks for cascaded GRU networks under the folder are listed as:
```bash
1. 'GRU_model_same_seq_125_250.pkl'.
2. 'GRU_model_same_seq_250_500.pkl'.
3. 'GRU_model_same_seq_500_950.pkl'.
4. 'GRU_model_same_seq_250_coe_fine_tune_all.pkl'.
```

The script ```landau_polariton_demo.py``` will process the test dataset, load the trained model, and predict both time-domain waveforms and resonance frequencies simultaneously. Three outputs will be generated: one sample predicted vs target time-domain waveform plot (saved as ```sample_raw_prediction_waveform.pdf```), one predicted vs target resonance frequencies (saved as ```coefficient_prediction_waveform.pdf```), one fft processed plot of the zero-padded input, predicted, target time-domain waveforms (saved as ```waveform_fft_result.pdf```).
All the raw data (target and prediction) are saved in the file ```output_landau_polariton.mat```.
The three models  ```GRU_model_same_seq_125_250.pkl```,  ```GRU_model_same_seq_250_500.pkl```,  ```GRU_model_same_seq_500_950.pkl``` are connected in a cascaded way to use the input data with 125 sequence length to predict the next 825 sequence points.

Under ```input_sequence``` folder, the script ```GRU_wave_input_sequence_test.py``` will generate one example predicted waveform figure (```input_seq_xx_result.pdf```) along with a .mat (```input_seq_xx_result.mat```) file that contains the complete predicted results and target results with difference input sequence lengths. 

Under ```noise``` folder, the script ```noise_demo.py``` will generate one example predicted waveform figure (```A_xxsample_result.pdf```) along with a .mat (```A_xxsample_result.mat```) file that contains the complete predict results and target results with difference noise strength A (from 0.01 to 0.2).

Under ```sampling``` folder, the script ```sampling_test.py``` will generate one example predicted waveform figure(```sampling_xx_result.pdf```) along with a .mat (```sampling_xx_result.mat```) file that contains the complete predict results and target results with difference sampling gap (from 0.5 to 20).

Under ```shuffle``` folder, the script ```shuffle_test.py``` will train 38 different models with randomly shuffled training data and will generate the complete training loss and test loss stored in ```shuffle_data_eval_loss.mat```.

Under ```visualization``` folder, the script ```visualization_demo.py``` will generate the plot of the visualization of the hidden state of our model. It will be saved in ```hidden_state_visulization.pdf```. Each run could generate different visualization plots. 

installation time: < 30s.
running time (for metamaterials): < 30s (with GTX1080 GPU), >10min (with CPU).
running time (others): < 30s.
