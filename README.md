# Chess

CNN model that attempts to classify whether a chess board state is invalid or not.

Setup Options:
 
### Option 1. Miniconda/Pycharm

- [Download miniconda](https://docs.anaconda.com/miniconda/)
- Create Virtual Enviroment

   1. Open terminal
   2. `conda create --name myenv python=3.12 # "myenv" can be whatever you want` 
   3. `conda activate myenv # activates virtual env`

- Configure python interpreter 

  1. If on pycharm community edition, go to `Settings> Python Interpreter> Add interpreter> Add Local Interpreter> Conda Enviroment > Use Existing Enviroment`. Your conda.exe file should be something like `/Users/bende/miniconda3/condabin/conda`

  2. `Use existing enviroment` - name of your virutal enviroment and then click `Ok`
  3. You should see your enviroment name in the lower right hand corner
  4. Open terminal in pycharm
  5. `conda activate yourenviromentname` and `pip install -r requirements.txt`
    
### Option 2. Miniconda/VSCode

   1. Install the VSCode Python Extension
     
   2. `Ctrl + Shift + P` on Windows or `Cmd + Shift + P` on Mac, type `Select Interpreter` and press Enter
   
   3. Select your virtual enviroment you made
     
   4. Open a terminal and run `conda activate yourenviromentname` and `pip install -r requirements.txt`

### Option 3. Terminal Only

   1. Create a virtual environment with `py -m venv yourenvironmentname`

   2. Activate the environment with `.\yourenvironmentname\Scripts\activate` on Windows or `source yourenvname/bin/activate` on Mac/Linux

   3. Install dependencies with `pip install -r requirements.txt`
