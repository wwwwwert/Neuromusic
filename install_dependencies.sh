sudo apt update
sudo apt install fluidsynth
sudo gem install yadisk

pip install -r requirements.txt

# Load pretrained models
yadisk https://disk.yandex.ru/d/6Q4qPGnUyCcvxg
unzip models.zip
cd models/

cd REMI
unzip \*.zip
cd ..

cd Structures
unzip \*.zip
cd ..

cd TSD
unzip \*.zip
cd ..
