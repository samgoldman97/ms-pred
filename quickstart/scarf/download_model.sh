wget https://www.dropbox.com/scl/fi/l3t1dmxgazqji9yoe98g9/canopus_scarf_models.zip?rlkey=zmkiz5c6kgt8qpm0o8z8qyt2a -O canopus_scarf_models.zip

unzip canopus_scarf_models.zip

mkdir quickstart/scarf/models

#mv nist_thread_model.ckpt quickstart/scarf/models/ 
#mv nist_weave_model.ckpt quickstart/scarf/models/ 

mv canopus_thread_model.ckpt quickstart/scarf/models/ 
mv canopus_weave_model.ckpt quickstart/scarf/models/ 

rm canopus_scarf_models.zip
