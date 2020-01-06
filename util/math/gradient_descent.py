import numpy as np

EPSILON = 1e-5


# snapshot of one gradient descent iteration
class GradientDescentSnapshot:

    def __init__(self, iteration_num, cost_value, loss_value, theta):
        self.iteration_num = iteration_num
        self.cost_value = cost_value
        self.loss_value = loss_value
        self.theta = theta


# container to contain result of gradient descent execution
class GradientDescentResult:

    def __init__(self):
        self.snapshots = []

    def has_snapshots(self):
        return len(self.snapshots) > 0

    def save_snapshot(self, i, cost, loss, theta):
        self.snapshots = self.snapshots.append(GradientDescentSnapshot(i, cost, loss, theta))


# configuration to run gradient descent
class GradientDescentConfiguration:

    def __init__(self, loss_func, hypothesis_func, learning_rate, lambda_value, is_save_snapshots):
        self.loss_func = loss_func
        self.hypothesis_func = hypothesis_func
        self.learning_rate = learning_rate
        self.lambda_value = lambda_value
        self.is_save_snapshots = is_save_snapshots


# --- loss functions ---
def linear_loss(h, y):
    """
    Linear loss for gradient descent
    :param h: value which returns hypothesis
    :param y: output values
    :return: linear loss
    """
    return h - y


def logistic_loss(h, y):
    """
    Logistic loss for gradient descent
    :param h: result which returns hypothesis function
    :param y: output values
    :return: logistic loss
    """
    return (-y * np.log(h + EPSILON) - (1 - y) * np.log(1 - h + EPSILON)).mean()


# --- gradient descent ----
def gradient_descent(features, results, gd_config):
    """
    Gradient descent realization that accepts features, results (Xs and Ys) and configuration for gradient descent
    :param features: Xs
    :param results: Ys
    :param gd_config: configuration to run gradient descent
    :return: packed gradient descent result, which has configured thetas and can has snapshots of execution
    """
    m = len(features)
    thetas = np.ones()
    gd_result = GradientDescentResult()

    #for i in range(gd_config)