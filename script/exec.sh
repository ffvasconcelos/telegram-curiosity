echo "Install dependencies"
pip install datetime
pip install numpy
pip install pandas
pip install networkx
pip install scipy
pip install matplotlib
pip install biokit
pip install statsmodels
pip install tqdm
pip install numba==0.50.1

echo "-------------------------------------------------------------"
echo "Running script 1"
python3 ./PCurioModelLinearZap.py

echo "-------------------------------------------------------------"
echo "Running script 2"
python3 ./DataAnalysisZap.py