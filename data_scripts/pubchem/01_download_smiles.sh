# Download CID-SMILES
wget https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz

# Unzip
gunzip CID-SMILES.gz

mkdir data/retrieval
mkdir data/retrieval/pubchem
mv CID-SMILES data/retrieval/pubchem/pubchem_full.txt
