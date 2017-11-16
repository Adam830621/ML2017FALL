echo "Downloading model"

wget --no-check-certificate 'https://www.dropbox.com/s/zp0bmrgyxss7faq/model_00027_0.68896.h5?dl=0' -O model_00027_0.68896.h5


echo "run"
python3 hw3_test.py $1 $2
