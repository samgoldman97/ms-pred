#wget ??nist_iceberg_models.zip

wget https://www.dropbox.com/s/omark23l0ee22hf/canopus_iceberg_models.zip
unzip canopus_iceberg_models

mkdir quickstart/iceberg/models

#mv nist_iceberg_generate.ckpt quickstart/iceberg/models/ 
#mv nist_iceberg_score.ckpt quickstart/iceberg/models/ 

mv canopus_iceberg_generate.ckpt quickstart/iceberg/models/ 
mv canopus_iceberg_score.ckpt quickstart/iceberg/models/ 

#rm nist_iceberg_models.zip
rm canopus_iceberg_models.zip