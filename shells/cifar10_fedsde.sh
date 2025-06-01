
nc=5 # number of clients
dda=0.6 # dirichlet dataset alpha
md=Conv4 # Conv5 or ResNet18
cuda=1 # cuda id

python fedsde.py \
    -c configs/cifar10/fedsde.yaml \
    -dda $dda \
    -md $md \
    -is 42 \
    -sn FedSDE_dir${dda}_nc${nc} \
    -g $cuda \
    -nc $nc \
 