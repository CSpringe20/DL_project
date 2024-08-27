CNN.py -> Structure and training + saving of the  discriminative CNN (generated-original data) -- change references before run
diffuser.py -> Code to generate DM dataset -- executed on multiples devices and then collected all toghether the datas since the process is very expensive
ident.py -> Code to count the number of identical files in a directory (and subdirectories) -- change references before run
res.txt -> Output of the latest training saved in run directory
save.py -> Code to download CIFAR dataset as png files
models -> Models of the training(s)
runs -> Log details from SummaryWriter (from torch.utils.tensorboard)
data.zip -> Both generated with DM and original dataset
