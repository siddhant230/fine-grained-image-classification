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

for Resnet-18 as baseline model

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
