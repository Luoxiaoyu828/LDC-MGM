## 0.0 Check Anaconda installation 
Firstly, make sure your `conda` command is available in your current shell terminal.
Then use the following commands to check the root path of the anaconda installation.
```bash
conda activate  base # use base environment
source activate

echo $_CONDA_ROOT # this command shall print the root path of the anaconda installation
echo $Env:_CONDA_ROOT # for windows power shell
```
返回的结果为anaconda的路径, 确认返回值不为空且确实为想要使用的anaconda路径，转到下一步

## 0.1 Check files in use

The following files are required:
```
<Where you have unzip the files>
├── ConBased-0.0.3.tar.gz
├── DensityClust-1.3.0.tar.gz
├── FacetClumps-0.2.tar.gz
├── README.md
├── config_init.bat
├── config_init.ps1
├── config_init.sh
├── detect_main.bat
├── detect_main.ps1
├── detect_main.sh
└── requirement.txt
```


## 1. Install Python and Python packages.
(Please replace `<env-name>` with the environment name you prefer) 

```bash
conda create -n <env-name> python=3.8
conda activate <env-name>
cd <Where you have unzip the files>
pip install -r requirement.txt
pip install DensityClust-1.3.0.tar.gz
pip install FacetClumps-0.2.tar.gz
pip install ConBased-0.0.3.tar.gz
```

## 2. Copy the scripts into environment 
```bash
conda activate <env-name>
cp *.sh $CONDA_PREFIX/bin/   # for unix-like systems
cp *.bat %CONDA_PREFIX%\Scripts\ # for cmd in Windows
cp *.ps1 $Env:CONDA_PREFIX\Scripts\ # for windows powershell
```

## 3. Change to a working directory and be sure to activate the environment before starting
```bash
cd <Where you want to work>
conda activate <env-name>
```

##  4. Generate default configuration file
Use the following command:
```bash
config_init
```
Then make sure that there is a file named 'config.yaml' in the working directory
```
vim config.yaml # edit the config file with vim (or any other editor you prefer)
```
Change the parameters as needed (short descriptions for the parameters are provided in `config.yaml`), then execute the detection command:
```
detect_main
```

## Upgrade

Part of the package can be upgraded by 

```bash
conda activate <env-name>
pip install --upgrade --force-reinstall <Name-of-Package>
```

<!-- ```ps
gci env:* | sort-object name
```
```ps
$Env:_CONDA_ROOT
```
 -->




