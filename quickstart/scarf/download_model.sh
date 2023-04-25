
#wget https://www.dropbox.com/s/s6gzr1mo9il5lbu/nist_scarf_models.zip
#wget https://www.dropbox.com/s/s6gzr1mo9il5lbu/nist_scarf_models.zip

wget https://www.dropbox.com/s/6ibgsciqpyeuaw7/canopus_scarf_models.zip
unzip canopus_scarf_models

mkdir quickstart/scarf/models

#mv nist_thread_model.ckpt quickstart/scarf/models/ 
#mv nist_weave_model.ckpt quickstart/scarf/models/ 

mv canopus_thread_model.ckpt quickstart/scarf/models/ 
mv canopus_weave_model.ckpt quickstart/scarf/models/ 

rm canopus_scarf_models.zip
