## flir_adas

refer: https://adas-dataset-v2.flirconservator.com/#downloadguide

```
for i in {00..11}; do curl --output FLIR_ADAS_v2.zip.${i} https://adas-dataset-v2.flirconservator.com/dataset/parts/FLIR_ADAS_v2.zip.${i}; done
``` 


Combine (concatenate) the files


   `cat FLIR_ADAS_v2.zip.* > FLIR_ADAS_v2.zip`


Extract the zip archive. Use your preferred method. You can use the following terminal command:

   `unzip -q FLIR_ADAS_v2.zip`


## hscai
```
mkdir hscai
cd hscai
wget https://61d600ca-25df-43e7-85db-3cbd0ac09ec1.filesusr.com/ugd/373ade_eaebeb24d0744c97a2782111bf73ba45.json?dn=train.json -O train.json
wget https://61d600ca-25df-43e7-85db-3cbd0ac09ec1.filesusr.com/ugd/373ade_0dcab4babd9844fd8cd4153574898645.json?dn=val.json -O val.json

cd ../..
python converter.py

cd datasets/hscai
wget https://www.hscaichallenge.com/_files/archives/373ade_453e36444dec4e869a46dd6d3ae76331.zip?dn=train.zip -O train.zip
wget https://www.hscaichallenge.com/_files/archives/373ade_36821d1f62744a769dd6bf11d1d03da9.zip?dn=val.zip -O val.zip
wget https://www.hscaichallenge.com/_files/archives/373ade_92fecc2d5eac467096afb17b04db3116.zip?dn=test_open.zip -O test_open.zip
unzip val.zip -d val/
unzip train.zip -d train/
unzip test_open.zip -d test_open/
rm val.zip 
rm train.zip 
rm test_open.zip 
```