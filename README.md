# Family Power Prediction

This repository contains code for predicting family power using machine learning techniques.


## LSTM

train

```python
python main.py -t -m lstm
```

evaluate

```python
python main.py -m lstm -l results/lstm-90-90-512-1-0.03.pth
```

or run test for task 1 in 90 days and 365 days
```python
python test.py
```

results
90 days for 5 per experiment results

```text
Task 1 (90 days) results: MSE = 0.0027 ± 0.0001, MAE = 0.0404 ± 0.0005
```

365 days for 5 per experiment results

```text
Task 1 (365 days) results: MSE = 0.0037 ± 0.0001, MAE = 0.0475 ± 0.0005
```
