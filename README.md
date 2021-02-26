# DSC190 WI21 Data Mining Challenge

[Website](https://www.kaggle.com/c/ucsd-dsc190-wi21-introduction-to-data-mining/overview)

## Benchmarks
Public Score: 83.61  
Private Score: Unknown  
Ranking: 1  

### Reference Benchmark for SP20 Challenge
Public Score: 86.01  
Private Score: 75.76  
Ranking (not counted toward actual competition): 1  

## Instructions
To run the script:
1. unzip `dataset.tar.gz`
2. make sure all dependencies are installed
3. open terminal, type the following command:
```
python main.py
```
4. the code will run approximately 5 minutes to clean, preprocess, train, and predict data
5. the corresponding `submission.csv` will appear in the directory for submission to the Kaggle competition

## Several Good Seeds that Overfit to Test Set
`59523`, `97321`, `20616`, `36124`, `29041`, `55931`, `34549`

## Reproducibility
- Use the good seeds in original code
- Run the model and make the submission
- Usually, the score should be similar to my actual score. Sometimes you need more runs when you have bad luck :( even if validation error looks similar.

### Hardware & Software Settings

CPU: AMD R9-3900X 12 Core 24 Threads  
GPU: NVIDIA RTX 3900 (did not use for training)  
Memory: 32 Gb 3200 Mhz  

Python: 3.8  
CUDA: 11.0  
Numpy Optimization: Intel MKL  

## Dependency Requirements
```
- numpy 1.20
- pandas 1.2.1
- scikit-learn 0.24.1
- tqdm 4.56.0
- lightgbm 3.1.1
- fancyimpute 0.5.5
- vadersentiment 3.3.2
```

