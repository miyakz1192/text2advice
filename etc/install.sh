set -x

cd /etc/systemd/system 
sudo rm text2advice.service
sudo ln -s ~/text2advice/etc/text2advice.service text2advice.service

sudo systemctl enable text2advice.service

