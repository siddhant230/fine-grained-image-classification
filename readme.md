# Baseline for Fine-grained Image Classification

Data pulled from here: https://github.com/AndresPMD/Fine_Grained_Clf

Data sample
<p align="left">
            <img src="outputs/data.png" alt="data"/>
</p> 

Data label distribution
<p align="left">
            <img src="outputs/data_distribution.png" alt="data distribution"/>
</p> 

results

for Frozen pre-trained model

```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00       198
           1       0.00      0.00      0.00       218
           2       0.00      0.00      0.00       200
           3       0.00      0.00      0.00       199
           4       0.00      0.00      0.00       213
           5       0.00      0.00      0.00       189
           6       0.00      0.00      0.00        29
           7       0.06      0.64      0.10       204
           8       0.00      0.00      0.00       215
           9       0.00      0.00      0.00       187
          10       0.00      0.00      0.00       198
          11       0.00      0.00      0.00       176
          12       0.00      0.00      0.00       223
          13       0.05      0.06      0.05       208
          14       0.00      0.00      0.00       190
          15       0.00      0.00      0.00       192
          16       0.06      0.27      0.09       211
          17       0.00      0.00      0.00       197
          18       0.00      0.00      0.00       174
          19       0.00      0.00      0.00        76

    accuracy                           0.05      3697
   macro avg       0.01      0.05      0.01      3697
weighted avg       0.01      0.05      0.01      3697

```


for simple custom classifier
```
              precision    recall  f1-score   support

           0       0.81      0.83      0.82       198
           1       0.84      0.84      0.84       218
           2       0.82      0.82      0.82       200
           3       0.75      0.78      0.76       199
           4       0.65      0.80      0.71       213
           5       0.82      0.85      0.84       189
           6       0.96      0.90      0.93        29
           7       0.75      0.81      0.78       204
           8       0.88      0.76      0.81       215
           9       0.84      0.76      0.80       187
          10       0.77      0.79      0.78       198
          11       0.85      0.81      0.83       176
          12       0.92      0.80      0.86       223
          13       0.78      0.80      0.79       208
          14       0.87      0.75      0.81       190
          15       0.84      0.82      0.83       192
          16       0.82      0.81      0.81       211
          17       0.79      0.78      0.78       197
          18       0.66      0.82      0.73       174
          19       0.98      0.78      0.87        76

    accuracy                           0.80      3697
   macro avg       0.82      0.80      0.81      3697
weighted avg       0.81      0.80      0.80      3697

```

for simple classifier with batch norms and more parameters

