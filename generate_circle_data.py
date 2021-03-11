import numpy as np
import matplotlib.pyplot as plt


class circle_data:
    """
    Generates random x,y coordinates, and classifies them according to a user specified radius.
    """

    def __init__(self, x_range=1, y_range=1, class_radius=0.5, n=500):
        """
        :param x_range: denotes range for uniform random sampling, running from -x_range to +x_range
        :param y_range: just like x_range, but for y
        :param class_radius: set the radius which discriminates between two classes
        :param n: number of points to generate
        """
        self.x_range = x_range
        self.y_range = y_range
        self.radius = class_radius
        self.n = n

        """
        generate the random points. Note that the explicit loop isn't ideal, but given the fact we need to classify
        each point, we'd have to iterate over whatever vectors we generated anyway. This should be a fairly low cost
        operation anyway.        
        """
        x_data = []
        y_data = []
        classification = []
        coords=[]
        for _ in range(n):
            x_data.append(np.random.uniform(low=-x_range, high=x_range))
            y_data.append(np.random.uniform(low=-y_range, high=y_range))
            coords.append((x_data[-1],y_data[-1]))
            if x_data[-1] ** 2 + y_data[-1] ** 2 > self.radius ** 2:
                classification.append(1)
            else:
                classification.append(0)
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.coords=coords
        self.targets= np.array(classification)

    def __repr__(self):
        return 'Circle data, n={}, x_range=[-{},{}],y_range=[-{},{}],classification radius={}'.format(self.n,
                                                                                                      self.x_range,
                                                                                                      self.x_range,
                                                                                                      self.y_range,
                                                                                                      self.y_range,
                                                                                                      self.radius)
    def plot(self):
        plt.scatter(self.x_data,self.y_data)
        circle = plt.Circle((0, 0), self.radius, color='r',fill=False,linewidth=3.0)
        plt.gcf().gca().add_artist(circle)
        plt.gcf().gca().set_aspect(1)
        plt.tight_layout()
        plt.show()

# a=circle_data(1,1,0.5,1000)
# print(a)
# a.plot()