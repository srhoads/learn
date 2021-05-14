# `learn`
This repo has been rendered for the initial purpose of introducing Arend M. W. to the coding world.

## SETUP!

## Step 1!
- Make sure `git` is installed
```sh
## make sure u have homebrew: 
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git
```

## Step 2!
- Install & setup Python3 (the right way)

```sh
python -V # check the version
```

- I recommend `pyenv`...
```sh
brew install pyenv
pyenv install 3.8.7
pyenv global 3.8.7
pip install ipython # source ~/.bash_profile
```

## Step 3!
- Install & setup your IDE(s)/Text editor(s)/etc...
    + VSCode: https://code.visualstudio.com/download
    + Jupyter
    ```sh
    pip install jupyterlab
    ```





------------------------------------------------------------------
# More...
------------------------------------------------------------------

### Using `git`!
- Clone this repository
- Learn how to read the data: `data/nrg_fng_04022019.f` (hint: file type is of `feather` format)
- Commit, push, and pull changes!


### Step 2!
#### Predict someone's gender from their name (using `Python`)
- Look into `Python` machine learning classification algorithms, install the relevant packages, get familiar with hyperparameters
- Train a basic model (with data provided by Sam). Explore its accuracy. Test it on some names.
- Try grid searching or randomized hyperparameter tuning (hint: `GridSearchCV` & `RandomSearchCV`)
- Also consider different ways of pre-processing your data & feature engineering before training some models (hint: `CountVectorizer` can be helpful!)

### Step 3!
- TBD... stay tuned!!


DEBUGGING NOTES:
We noticed you're using a conda environment. If you are experiencing issues with this environment in the integrated terminal, we recommend that you let the Python extension change "terminal.integrated.inheritEnv" to false in your user settings.

For compilers to find readline you may need to set:
  export LDFLAGS="-L/usr/local/opt/readline/lib"
  export CPPFLAGS="-I/usr/local/opt/readline/include"

For pkg-config to find readline you may need to set:
  export PKG_CONFIG_PATH="/usr/local/opt/readline/lib/pkgconfig"