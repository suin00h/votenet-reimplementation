class_list = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
              'bottle', 'bowl', 'car', 'chair', 'cone',
              'cup', 'curtain', 'desk', 'door', 'dresser',
              'flower', 'glass', 'guitar', 'keyboard', 'lamp',
              'laptop', 'mantel', 'monitor', 'night', 'person',
              'piano', 'plant', 'radio', 'range', 'sink',
              'sofa', 'stairs', 'stool', 'table', 'tent',
              'toilet', 'tv', 'vase', 'wardrobe', 'xbox']

def get_cls_idx(cls: str):
    return class_list.index(cls)

def get_cls(idx: int):
    return class_list[idx]