
wget https://www.dropbox.com/s/s6gzr1mo9il5lbu/nist_scarf_models.zip
unzip nist_scarf_models

mkdir quickstart/models

mv nist_thread_model.ckpt quickstart/models/ 
mv nist_weave_model.ckpt quickstart/models/ 

rm nist_scarf_models.zip
