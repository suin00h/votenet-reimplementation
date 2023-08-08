import matplotlib.pyplot as plt

def show_cloud(cloud):
    x = cloud[:,0].squeeze()
    y = cloud[:,1].squeeze()
    z = cloud[:,2].squeeze()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    ax.scatter3D(x, y, z, marker='o')
    plt.show()