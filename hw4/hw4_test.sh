echo "---------Downloading model--------"
wget --no-check-certificate 'https://www.dropbox.com/s/pgtq83j2xk4a1yt/emb2.zip?dl=0' -O emb2.zip
unzip emb2.zip
rm emb2.zip
echo "--------Start testing---------"
python3 hw4_test.py $1 $2