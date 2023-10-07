LINK=https://www.dropbox.com/scl/fi/13yt9gozdr6yf3jl908u8/canopus_train_public.zip?rlkey=lnfwjxutp90zqo89gfj7m15ot


wget "$LINK" -O canopus_train_public.zip

# Unzip the archive
unzip canopus_train_public.zip

# Delete the original ZIP archive
rm canopus_train_public.zip

# Move extracted data to the target directory
mv canopus_train_public data/spec_datasets/
