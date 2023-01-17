#!/bin/bash
cd ~/Desktop/ma/spektral
sudo python setup.py install
#python setup.py install

cd -
python glrp.py -c ~/Desktop/ma/gcnn-and-grlp/glrp/config/test.json --test-seed 42 -t
