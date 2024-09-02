## flir_adas

refer: https://adas-dataset-v2.flirconservator.com/#downloadguide

```for i in {00..11}; do curl --output FLIR_ADAS_v2.zip.${i} https://adas-dataset-v2.flirconservator.com/dataset/parts/FLIR_ADAS_v2.zip.${i}; done``` 
Combine (concatenate) the files
   `cat FLIR_ADAS_v2.zip.* > FLIR_ADAS_v2.zip`
Extract the zip archive. Use your preferred method. You can use the following terminal command:
   `unzip -q FLIR_ADAS_v2.zip`


## hscai
```
train: wget https://www.hscaichallenge.com/_files/archives/373ade_453e36444dec4e869a46dd6d3ae76331.zip?dn=train.zip
val: wget https://www.hscaichallenge.com/_files/archives/373ade_36821d1f62744a769dd6bf11d1d03da9.zip?dn=val.zip
test_open: wget https://www.hscaichallenge.com/_files/archives/373ade_92fecc2d5eac467096afb17b04db3116.zip?dn=test_open.zip
```