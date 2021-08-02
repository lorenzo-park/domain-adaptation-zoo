import torchvision.transforms as T

def get_transforms_officehome():
    """
    CREDITS: https://github.com/VisionLearningGroup/OVANet/blob/f6ca72b8ce760764e9d5252a0df8c6e7ca0ec2b3/utils/defaults.py
    """
    data_transforms = {
        "src": T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "tgt": T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms
