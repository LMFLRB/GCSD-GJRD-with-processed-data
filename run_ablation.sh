# for stl10
pythom -m \
    --dataset STL10 \
    --resnet_type  resnet50 \
    --feature_type linear \
    --loss GJRD \
    --select_version 2

# for fashion
pythom -m \
    --dataset FashionMNIST \
    --resnet_type  resnet18 \
    --feature_type linear \
    --loss GJRD \
    --select_version 0

    
# for mnist
pythom -m \
    --dataset MNIST \
    --resnet_type  resnet18 \
    --feature_type linear \
    --loss GJRD \
    --select_version 0