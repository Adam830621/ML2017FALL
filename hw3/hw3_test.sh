echo "---------Downloading model--------"
wget --no-check-certificate 'https://www.dropbox.com/s/wkfr7dslre3ytk6/model_00027_0.68896.zip?dl=0' -O model_00027_0.68896.zip
unzip model_00027_0.68896.zip
rm model_00027_0.68896.zip
echo "--------Start testing---------"
python3 hw3_test.py $1 $2
