# FedSDE
The official code repository for the submission ICDM paper

## Installation

### Dependencies

 - Python (3.9)
 - PyTorch (2.4.0)
 - numpy (1.24.3)

### Install requirements

Run the following command to install the required packages:

`pip install -r requirements.txt` 
## Run Code

For example, to reproduce the results of the CIFAR-10 dataset with alpha = 0.6 (Table 2) and 10 clients, you can use the following bash code
```bash
# bash
bash shells/cifar10_fedsde.sh
```

## Experimental Result
Table1.  Test accuracy of the server model of different methods over five datasets and across three levels of statistical heterogeneity
| Method      | $Dir(alpha)$ |Ensemble |CoBoosting |DENSE |FedSD2C |FedSDE |
| ----------- | ----------- |----------- |----------- |----------- |----------- |---------- |
| CIFAR-100 | $alpha$=0.1 |27.70|26.72|26.72|29.20|**30.12**| 
| | $alpha$=0.05 | 21.59|18.24|26.76|20.66|**27.87**|
| | $alpha$=0.01 | 16.39|17.03|16.87|19.76|**20.38**|
TINY-ImageNet | $alpha$=0.1 |11.03|16.03| 15.69 |17.02|**18.39**|
| | $alpha$=0.05 | 10.53|11.06|10.88|13.21|**14.18**|
| | $alpha$=0.01 |7.85 | 10.12|8.83|12.26|**12.81** |
 CIFAR-10 | $alpha$=0.1 |42.45|45.12|44.87|44.34|**45.27**|
| |$alpha$=0.05|28.54|31.94|30.26|43.26|**45.11**|
| |$alpha$=0.01|23.51|25.39|22.41|28.22|**29.02**|
Imagettee | $alpha$=0.1 |32.33|35.23|45.94|**49.01**|48.32|
| | $alpha$=0.05 |13.04|21.66|20.64|**23.14**|23.04|
| | $alpha$=0.01 |11.03|15.27|14.62|12.13|**16.12**|

Table2.Test accuracy of the server model of different methods over five datasets and across three levels of Client Number $C$

 |Method | $C$  | FedAvg | CoBoosting | DENSE | FedSD2C  | FedSDE | 
 | ----------- | ----------- |----------- |----------- |----------- |----------- |---------- |
CIFAR-100 | 5  | 36.00| 42.13 | 41.19|  45.34 |   **48.90**|
 | | 10 |36.16  | 37.20 | 37.23 | 38.79   | **39.03** |
 | | 20 | 30.35 | 29.11 | 32.76  |  32.10| **33.51**|
 |  | 50   | 20.24|25.62| 24.06   | 30.20  | **31.22**|
TINY-ImageNet  | 5  | 30.99 | 32.50 | 32.33 |  34.57 | **35.11**|
 | | 10 | 13.23 | 14.65 | 14.84 | 17.35  |  **18.94**|
 | | 20 | 10.23 | 11.86 | 12.87 | 15.33  |  **16.29**|
 |  | 50 | 6.87 | 8.12 | 7.59 | 10.77  | **12.01**|
 CIFAR-10  | 5 | 67.67  | 69.92 | 68.12 | 71.23  |  **72.84** |
 | | 10 | 60.53 | 62.11 | 61.80 | 67.10  |  **67.98** |
 | | 20 |  51.39 | 58.20 | 51.56 |   54.92|  **61.74** |
 |  | 50  | 42.69| 44.31| 44.11 |   50.98|  **55.27** |
 |Imagettee  | 5 | 63.34  | 67.21 | 65.24 | 67.62  |  **68.54** |
 | | 10 | 60.54| 64.43 | 64.28  |  **65.12** | 65.06  |
 | | 20 | 54.32| 56.05 | 55.34 | 56.78  |  **57.15** |
   || 50  | 44.92 |  **52.10**| 49.12 | 46.25  | 51.23  |
