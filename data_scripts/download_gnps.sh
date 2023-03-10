cd data/spec_datasets/
wget https://www.dropbox.com/s/8jn6sz0o3srmtev/canopus_train_public.tar
tar -xvf canopus_train_public.tar
rm canopus_train_public.tar
cd canopus_train_public
rm -rf sirius_outputs
rm -rf retrieval_hdf
rm -rf magma_outputs
rm -rf splits
cd ../../