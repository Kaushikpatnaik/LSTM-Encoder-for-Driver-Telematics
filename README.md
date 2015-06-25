This is a stab at solving the AXA Driver telematics challenge in Kaggle using deep learning Techniques. For the contest
information please see the link below.
https://www.kaggle.com/c/axa-driver-telematics-analysis

Approach:
At the outset, the problem definition is one of outlier detection. Like other teams, we have converted the outlier detection problem to a classification problem,
where for each trip we classify whether it belongs to the driver or not. Thus for each driver we run a binary classification problem taking 160 out of 200 provided 
trips as training, 20 as validation and 20 as testing. 10-50 trips were choosen from 100 other drivers. The dataset was kept unbalanced. 

Minimal feature generation steps were performed on the given data. We extracted the min, max and avg values of velocity, accelaration and direction change in a window of 4 secs. Therefore
for a trip that lasted 200 secs, would be a sequence of 50 feture input vectors for our model. Also, trips lengths were shortened to nearest multiples of 200 for ease of use while processing.

Model: 
Due to the sequential nature of the data, we used multi layered RNN/LSTMs to classify driver trips. The multi-layered LSTMs encoded the sequence information and only predicted the outout at the last time step. This is know as an Encoder-Decoder architecture described in
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
The models are thus both deep in time and deep in feature space. We restricted Backpropagation through time to 50 steps for compuational reasons. Input vectors were features such as acceleration, velocity, change of direction etc. The model output is be a soft max probability with criterion being cross-entropy error. There is regularization in the form of dropout both in time and in between levels. 

Implementation:
The model was implemented using Torch7, with a lot of the LSTM utilities being derived from 
https://github.com/oxford-cs-ml-2015/practical6
which showed a simple straightforward way to implement such models. We made changes to the code to accomodate encoder decoder forward and backpropagation, modified input streams, added code for 
splitting the dataset, added dropout to the model and other minor modifications.

Observation:
For a small sample subset provided in this repository we see that a 2 layer LSTM encoder classification approach achieves 89% accuracy without much feature engineering. Similar accuracies were
observed in our test set. We are currently in the process of submitting the complete results to the Kaggle leaderboard. We will update this page shortly based on the results.

# Running the system:
* Download the repository for github using
* Open Terminal and Run 'python preprocess_data.py'. This places the processed files in folder './proc_data'
* Run 'th loadData.lua -ipfilepath ./proc_data/classfile.csv -opfilepath ./torch_data/classfile.th7'. This loads the data in torch7 format for use in lua
* Run 'th trainDeepLSTM.lua -classfile ./torch_data/classfile.th7 -datafile ./torch_data/datafile.th7'

# Required Installations
Python==2.7
numpy==1.9.2

Lua and Torch7
# in a terminal, run the commands
curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; ./install.sh
source ~/.bashrc

Torch7 repositories:
luarocks install nn
luarocls install nngraph
luarocks install optim
luarocks math

