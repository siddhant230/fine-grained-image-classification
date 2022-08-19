# Baseline for Fine-grained Image Classification

Data pulled from here: https://github.com/AndresPMD/Fine_Grained_Clf

Data sample
<p align="left">
            <img src="imgs/data.png" alt="data"/>
</p> 

Data label distribution
<p align="left">
            <img src="imgs/data_distribution.png" alt="data distribution"/>
</p> 

results

for Resnet-18 as baseline model: splits

```
              precision    recall  f1-score   support

           0       0.63      0.68      0.65       443
           1       0.65      0.66      0.66       369
           2       0.90      0.86      0.88       412
           3       0.92      0.92      0.92       423
           4       0.45      0.79      0.57       218
           5       0.85      0.71      0.78        63
           6       0.61      0.35      0.44       147
           7       0.74      0.51      0.61       341
           8       0.49      0.62      0.54       138
           9       0.80      0.79      0.79       378
          10       0.70      0.56      0.62       111
          11       0.84      0.34      0.49        79
          12       0.77      0.68      0.72       418
          13       0.90      0.90      0.90       492
          14       0.59      0.52      0.55       352
          15       0.53      0.68      0.59       151
          16       0.81      0.69      0.74       303
          17       0.69      0.82      0.75       444
          18       0.83      0.86      0.85       425
          19       0.79      0.81      0.80       455

    accuracy                           0.73      6162
   macro avg       0.72      0.69      0.69      6162
weighted avg       0.75      0.73      0.73      6162
```



for Resnet-18 as baseline model: unrolled

```
              precision    recall  f1-score   support

           0       0.93      0.95      0.94       195
           1       0.93      0.93      0.93       194
           2       0.94      0.96      0.95       201
           3       0.94      0.96      0.95       208
           4       0.95      0.93      0.94       196
           5       0.97      0.96      0.97       226
           6       0.97      0.94      0.96        34
           7       0.99      0.93      0.96       202
           8       0.92      0.95      0.94       197
           9       0.91      0.94      0.92       204
          10       0.93      0.94      0.94       186
          11       0.95      0.94      0.95       178
          12       0.94      0.93      0.94       203
          13       0.97      0.94      0.95       195
          14       0.95      0.94      0.94       204
          15       0.95      0.93      0.94       190
          16       0.94      0.97      0.96       204
          17       0.93      0.97      0.95       218
          18       0.97      0.94      0.96       196
          19       0.95      0.92      0.94        66

    accuracy                           0.95      3697
   macro avg       0.95      0.94      0.95      3697
weighted avg       0.95      0.95      0.95      3697


```
