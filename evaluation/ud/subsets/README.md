# Subset Metadata

|set|language  |n_train_sents  |n_train_tokens  |n_dev_sents  |n_dev_tokens  |n_test_sents  |n_test_tokens  |avg_tokens_per_sentence|
| :---- | :------- | :------------ | :------------- | :---------- | :----------- | :----------- | :------------ | :---------------------- |
|primary| en       | 1196          | 20240          | 861         | 10747        | 894          | 10924         | 14.2023                 |
|primary| he       | 602           | 20216          | 362         | 10751        | 354          | 10924         | 31.7838                 |
|primary| zh       | 831           | 20229          | 434         | 10762        | 459          | 10949         | 24.3271                 |
|primary| vi       | 1400          | 20215          | 482         | 10755        | 746          | 10939         | 15.9471                 |
|primary| ko       | **1574**          | 20220          | 871         | 10752        | 881          | 10935         | **12.5998**                 |
|primary| tr       | **2178**          | 20219          | 1040        | 10757        | 1002         | 10922         | **9.92844**                 |
|primary| el       | 776           | 20246          | 403         | 10747        | 456          | 10922         | 25.6361                 |
|primary| id       | 915           | 20230          | 471         | 10750        | 515          | 10940         | 22.0516                 |
|primary| ja       | 862           | 20222          | 451         | 10786        | 454          | 10925         | 23.7312                 |
|secondary  |     fr|            830|           20261|          448|         10767|           416|          10295|                24.393743|
|secondary  |     fi|           1539|           20223|          817|         10757|           814|          10970|                13.233438|
|secondary  |     es|            612|           20235|          317|         10770|           337|          10965|                33.151659|
|secondary  |     fa|           1115|           20219|          622|         10765|           640|          10923|                17.630206|
|secondary  |     de|           1055|           20215|          682|         10756|           633|          10928|                17.678903|
|secondary  |     ru|           1173|           20236|          645|         10748|           608|          10939|                17.280709|
|secondary  |     hi|            972|           20216|          508|         10764|           529|          10934|                20.863116|

## Deprels

An alphabetically ordered list of the 37 deprels in UD:

['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']

`CHANGELOG.jsonl` contains history of the labels that were reduced for each set of each language.