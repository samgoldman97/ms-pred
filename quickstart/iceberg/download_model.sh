#wget ??nist_iceberg_models.zip

# wget https://www.dropbox.com/s/omark23l0ee22hf/canopus_iceberg_models.zip

# wget but send output to new file
wget https://www.dropbox.com/scl/fi/1g3d4mx6yrh5rkj56hucs/canopus_iceberg_models.zip?rlkey=be8h3m02jy4ondn5tr8sbcfsh -O canopus_iceberg_models.zip  


unzip canopus_iceberg_models.zip


mkdir quickstart/iceberg/models

#mv nist_iceberg_generate.ckpt quickstart/iceberg/models/ 
#mv nist_iceberg_score.ckpt quickstart/iceberg/models/ 

mv canopus_iceberg_generate.ckpt quickstart/iceberg/models/ 
mv canopus_iceberg_score.ckpt quickstart/iceberg/models/ 

#rm nist_iceberg_models.zip
rm canopus_iceberg_models.zip