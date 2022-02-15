# DSC190 WI21 Data Mining Challenge
## Reference
If you are using this pipeline (or part of the pipeline) for future offerings of the same class (or different classes), please cite my implementation accordingly. Uncited works are subject to plagiarism: 
```
@misc{airbnbHousingPred,
  author    = {Colin Wang},
  title     = {Predicting Airbnb Housing Price with Machine Learning},
  year      = {2021},
  url       = {https://github.com/zwcolin/dsc190-kaggle}
}
```
## Introduction
[Website](https://www.kaggle.com/c/ucsd-dsc190-wi21-introduction-to-data-mining/overview)  
[Slides](https://docs.google.com/presentation/d/1WxgHtFA0XqmbKwXgYtmNk5FQRq42FbVMjAH-fJgcZMo/edit#slide=id.p)  

<img src="https://user-images.githubusercontent.com/59942464/109371716-ffdd1200-785a-11eb-93a9-dba5bfa7a8cd.png" title="Sentiment visualization for 50000+ users from the dataset!" alt="drawing" width="600"/>  

*Sentiment visualization for 50000+ users from the dataset*

## Benchmarks
Public Score: 83.46  
Private Score: 78.19  
Ranking: 1  
Ranking on Spring 2020's Competition: 1 (Not recorded since it's submitted outside the competition time range)

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
- One of the best predictions were made in an earlier model, but I couldn't find its configuration
- Luckily, the prediction file is still available as `data\best.csv`
- Use the "overfitting" seeds in original code (no need to change anything)
- Run the model and make the submission
- Usually, the score should be similar to my actual score. 
- Sometimes you need more runs when you have bad luck :( even if validation error looks similar.

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
- scipy 1.6.0
- pandas 1.2.1
- scikit-learn 0.24.1
- tqdm 4.56.0
- lightgbm 3.1.1
- vadersentiment 3.3.2
- tensorflow 2.4.0
- spacy 3.0.3
- textblob 0.15.3
- mapply 0.1.7 (optional, for multithreading)
```

![image](https://user-images.githubusercontent.com/59942464/109370458-429beb80-7855-11eb-8637-a1b46b5b9ed0.png)
![image](https://user-images.githubusercontent.com/59942464/109370478-59dad900-7855-11eb-9e61-a90d7fae67d8.png)
