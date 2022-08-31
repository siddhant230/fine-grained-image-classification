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

OVERALL mAP
```
dataset | baseline | theirs | CLIP(zero-shot) | CLIP(trainable image) | CLIP(disjoint/OOD)-(zero-shot)
--------+----------+--------+-----------------+-----------------------+--------------------------------
bottle  | 63.70    | 73.41  |   47.88         |     20.46             |            50.57
context | 75.13    | 70.96  |   56.82         |     25.60             |            61.17
``` 

<table>
<tr>
   <th colspan="7">Image encoder</th>
   <th colspan="7">Method</th>
   <th colspan="14">Dataset</th>
</tr>

<tr>
   <th colspan="7"></th>
   <th colspan="3">ENCODER</th>
   <th colspan="3">NORMS</th>
   <th colspan="7">CONTEXT</th>
   <th colspan="7">BOTTLE</th>
</tr>

<tr>
   <th colspan="7" rowspan="3">VIT</th>
</tr>

<tr>
   <th colspan="3" style="font-size:15px;">frozen</th>
   <th colspan="3" style="font-size:15px;">frozen</th>
   <th colspan="7" style="font-size:15px;">83.20</th>
   <th colspan="7" style="font-size:15px;">69.20</th>
</tr>

<tr>
   <th colspan="3" style="font-size:15px;">frozen</th>
   <th colspan="3" style="font-size:15px;">train</th>
   <th colspan="7" style="font-size:15px;">88.05</th>
   <th colspan="7" style="font-size:15px;">73.70</th>
</tr>

<tr>
   <th colspan="7" rowspan="4">Res50</th>
</tr>
<tr>
   <th colspan="3">frozen</th>
   <th colspan="3">frozen</th>
   <th colspan="7">78.20</th>
   <th colspan="7">62.80</th>
</tr>

<tr>
   <th colspan="3">frozen</th>
   <th colspan="3">train</th>
   <th colspan="7">70.26</th>
   <th colspan="7">60.68</th>
</tr>
</table>

for CLIP (OOD)-(zero-shot):

****->OOD***

DATASET: BOTTLE

```
OOD : [1, 5, 6, 7, 8, 9, 12, 15, 16, 17]


              precision    recall  f1-score   support

          0       0.64      0.60      0.62       415
          1*      0.64      0.55      0.59       441
          2       0.90      0.94      0.92       412
          3       0.94      0.90      0.92       394
          4       0.47      0.40      0.43       185
          5*      0.49      0.94      0.64        72
          6*      0.22      0.41      0.29       146
          7*      0.49      0.49      0.49       368
          8*      0.24      0.40      0.30       157
          9*      0.77      0.83      0.80       401
          10      0.44      0.13      0.20       108
          11      0.16      0.49      0.25        76
          12*     0.60      0.11      0.18       385
          13      0.89      0.93      0.91       498
          14      0.44      0.52      0.48       325
          15*     0.30      0.38      0.33       165
          16*     0.65      0.66      0.66       324
          17*     0.76      0.46      0.57       487
          18      0.73      0.73      0.73       407
          19      0.62      0.77      0.68       397

    accuracy                           0.62      6163
   macro avg       0.57      0.58      0.55      6163
weighted avg       0.65      0.62      0.62      6163
```

DATASET: CONTEXT

```
OOD : [2, 5, 6, 11, 13, 15, 18, 19, 20, 21, 23, 25, 26, 27]


              precision    recall  f1-score   support

           0       0.83      0.84      0.83       406
           1       0.95      0.93      0.94       525
           2*      0.17      0.30      0.22        97
           3       0.78      0.94      0.85       445
           4       0.74      0.54      0.63       281
           5*      0.67      0.80      0.73        97
           6*      0.74      0.48      0.58       408
           7       0.54      0.66      0.59       380
           8       0.04      0.73      0.08        15
           9       0.92      0.83      0.88       399
          10       0.89      0.51      0.65       331
          11*      0.27      0.31      0.29        26
          12       0.71      0.82      0.76        94
          13*      0.56      0.85      0.68       403
          14       0.11      0.03      0.04       389
          15*      0.45      0.82      0.58       165
          16       0.94      0.80      0.86       208
          17       0.66      0.94      0.78       425
          18*      0.92      0.90      0.91       381
          19*      0.67      0.89      0.77       258
          20*      0.79      0.42      0.55       401
          21*      0.86      0.44      0.58       461
          22       0.60      0.85      0.70       110
          23*      0.53      0.44      0.48        78
          24       0.53      0.56      0.55       238
          25*      0.86      0.79      0.82       464
          26*      0.72      0.53      0.61       242
          27*      0.69      0.81      0.75       376

    accuracy                           0.69      8103
   macro avg       0.65      0.67      0.63      8103
weighted avg       0.72      0.69      0.69      8103
```


---

for CLIP(zero-shot):

DATASET: BOTTLE
```
              precision    recall  f1-score   support

           0       0.65      0.47      0.55       419
           1       0.55      0.55      0.55       390
           2       0.89      0.91      0.90       422
           3       0.97      0.82      0.89       411
           4       0.45      0.35      0.39       176
           5       0.52      0.56      0.54        57
           6       0.20      0.29      0.23       138
           7       0.42      0.42      0.42       338
           8       0.15      0.40      0.22       125
           9       0.79      0.76      0.78       427
          10       0.25      0.11      0.16       123
          11       0.15      0.63      0.24        86
          12       0.78      0.05      0.09       396
          13       0.82      0.92      0.87       490
          14       0.43      0.39      0.41       337
          15       0.19      0.63      0.30       166
          16       0.46      0.63      0.54       331
          17       0.87      0.22      0.35       447
          18       0.81      0.62      0.71       432
          19       0.70      0.71      0.70       452

    accuracy                           0.56      6163
   macro avg       0.55      0.52      0.49      6163
weighted avg       0.65      0.56      0.56      6163
```

DATASET: CONTEXT
```
               precision    recall  f1-score   support

           0       0.85      0.80      0.82       406
           1       0.96      0.77      0.85       525
           2       0.15      0.33      0.21        97
           3       0.80      0.91      0.85       445
           4       0.52      0.51      0.52       281
           5       0.41      0.63      0.50        97
           6       0.57      0.64      0.60       408
           7       0.68      0.53      0.60       380
           8       0.05      0.67      0.09        15
           9       0.77      0.78      0.78       399
          10       0.90      0.29      0.44       331
          11       0.52      0.54      0.53        26
          12       0.57      0.73      0.64        94
          13       0.59      0.85      0.70       403
          14       0.11      0.02      0.03       389
          15       0.34      0.77      0.47       165
          16       0.68      0.86      0.76       208
          17       0.61      0.89      0.72       425
          18       0.91      0.82      0.86       381
          19       0.75      0.67      0.71       258
          20       0.44      0.57      0.50       401
          21       0.94      0.30      0.46       461
          22       0.55      0.58      0.57       110
          23       0.38      0.42      0.40        78
          24       0.48      0.37      0.42       238
          25       0.70      0.75      0.72       464
          26       0.80      0.28      0.42       242
          27       0.73      0.80      0.76       376

    accuracy                           0.63      8103
   macro avg       0.60      0.61      0.57      8103
weighted avg       0.67      0.63      0.62      8103
```

for Resnet-18 as baseline model: splits

DATASET: BOTTLE
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

DATASET: CONTEXT
```

              precision    recall  f1-score   support

           0       0.96      0.85      0.90       404
           1       0.86      0.93      0.90       524
           2       0.62      0.98      0.76        95
           3       0.89      0.93      0.91       444
           4       0.75      0.70      0.73       279
           5       0.92      0.85      0.89        95
           6       0.85      0.80      0.83       408
           7       0.88      0.87      0.87       378
           8       0.67      1.00      0.80        14
           9       0.71      0.84      0.77       398
          10       0.87      0.92      0.89       330
          11       0.73      0.92      0.81        24
          12       0.94      0.87      0.90        92
          13       0.82      0.92      0.87       403
          14       0.86      0.80      0.83       389
          15       0.87      0.87      0.87       165
          16       0.96      0.91      0.93       208
          17       0.93      0.78      0.85       425
          18       0.95      0.86      0.90       380
          19       0.95      0.81      0.88       256
          20       0.89      0.78      0.83       400
          21       0.88      0.87      0.88       461
          22       0.59      0.95      0.73       108
          23       0.91      0.76      0.83        78
          24       0.83      0.86      0.85       236
          25       0.96      0.94      0.95       464
          26       0.81      0.74      0.78       242
          27       0.83      0.94      0.88       376

    accuracy                           0.86      8076
   macro avg       0.85      0.87      0.85      8076
weighted avg       0.87      0.86      0.86      8076
```

