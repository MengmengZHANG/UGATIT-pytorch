pip install opencv-python --user
pip install tensorflow==1.13.1 --user

x-small architecture: https://docs.google.com/spreadsheets/d/1-Iw7YDFJ150s7TTygnStCL7RZnNKaH6TFhm7LSyBSww/edit?usp=sharing

# connection
ssh -i ~/git-clones/AWS_P0.pem -L 8888:localhost:8888 ubuntu@3.91.30.108
ssh -i ~/git-clones/AWS_P0.pem ubuntu@3.91.30.108
# mount volume
lsblk 
cd /
sudo mkdir data
sudo file -s /dev/nvme2n1
# (only needed when the prev command show data) sudo mkfs.ext4 -E nodiscard /dev/nvme1n1
sudo mount /dev/nvme2n1 /data
df -h
sudo umount /data

# tmux
tmux new -s s0
Ctrl-b d
tmux attach -t s0
tmux kill-session -t s0
Ctrl-b % (new pane)
Ct%rl-b " (new pane)
Ctrl-b o (use to traverse panes)
Ctrl-b [ (scroll)

# git data
scp -i ~/git-clones/AWS_P0.pem -r ~/.ssh/id_rsa.pub ubuntu@54.196.221.145:~/.ssh/
scp -i ~/git-clones/AWS_P0.pem -r ~/.ssh/id_rsa  ubuntu@54.196.221.145:~/.ssh/
git pull

# kaggle data
pip3 install kaggle --user
scp -i ~/git-clones/AWS_P0.pem -r ~/.kaggle ubuntu@54.196.221.145:~/.kaggle
chmod 700 ~/.kaggle/kaggle.json 
kaggle competitions download -c 11785-fall19-hw4p2
unzip 11785-fall19-hw4p2.zip -d 11785-fall19-hw4p2

# train
source activate pytorch_p36
jupyter notebook --no-browser --port=8888
pip3 install gpustat --user
watch gpustat

python main.py --dataset selfie2anime_64_64 --resume True --batch_size 1 --iteration 72500

python main.py --dataset selfie2anime_64_64 --phase test



small
------------------------------------------------------------------
batch 16 => 32h
[40/10000] time: 75.3895 d_loss: 4.34614706, g_loss: 1579.65954590

batch 8 => 44h
[ 1000/10000] time: 1271.1577 d_loss: 3.51449919, g_loss: 1261.32763672
[ 1001/10000] time: 1274.3142 d_loss: 3.36542392, g_loss: 1292.40014648
[ 1002/10000] time: 1275.5625 d_loss: 3.24884176, g_loss: 1243.78735352
[ 1003/10000] time: 1276.8097 d_loss: 3.33150482, g_loss: 1222.08227539
[ 1004/10000] time: 1278.0559 d_loss: 3.71317768, g_loss: 1193.90466309
[ 1005/10000] time: 1279.3023 d_loss: 3.34724402, g_loss: 1248.31665039



xsmall
------------------------------------------------------------------
xsmall batch 16 => 17h
[  500/10000] time: 516.8388 d_loss: 3.46146202, g_loss: 1369.67602539
[  501/10000] time: 519.2474 d_loss: 3.26977706, g_loss: 1378.46948242
[  502/10000] time: 520.2820 d_loss: 3.54301906, g_loss: 1350.86083984
[  503/10000] time: 521.3176 d_loss: 3.43841100, g_loss: 1342.41674805
[  504/10000] time: 522.3537 d_loss: 3.42271471, g_loss: 1353.86059570

xsmall batch 8 => 19h
[ 1000/10000] time: 592.9606 d_loss: 3.51439667, g_loss: 1317.12280273
[ 1001/10000] time: 594.5407 d_loss: 3.52841067, g_loss: 1341.52172852
[ 1002/10000] time: 595.1436 d_loss: 3.80192089, g_loss: 1514.29809570
[ 1003/10000] time: 595.7344 d_loss: 3.31296873, g_loss: 1308.24365234
[ 1004/10000] time: 596.3323 d_loss: 3.20091319, g_loss: 1299.53234863
[ 1005/10000] time: 596.9279 d_loss: 3.38806343, g_loss: 1427.98217773
[ 1006/10000] time: 597.5213 d_loss: 3.19738626, g_loss: 1318.87695312

