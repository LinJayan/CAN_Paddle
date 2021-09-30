export PATH="~/anaconda4/bin:$PATH"
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
gunzip reviews_Books.json.gz
gunzip meta_Books.json.gz
python process_data/process_data.py meta_Books.json reviews_Books.json
python process_data/local_aggretor.py
python process_data/split_by_user.py
python process_data/generate_voc.py
