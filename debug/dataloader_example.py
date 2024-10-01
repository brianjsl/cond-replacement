from datasets.numerical.function1d import LinearNoisy
from omegaconf import DictConfig
import yaml
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = 'configurations/dataset/numerical_linear.yaml'
    with open(path) as stream:
        conf = yaml.safe_load(stream)
    data = LinearNoisy(DictConfig(conf), split='train')()
    plt.scatter(data['xs'], data['ys'])
    plt.xlim(0, 1)
    plt.ylim(0, 1.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Points in 2D')
    plt.show()
