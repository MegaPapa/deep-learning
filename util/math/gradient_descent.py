import numpy as np
import logging

EPSILON = 1e-5


class GradientDescentSnapshot:
    """
    Snapshot of one gradient descent iteration
    """

    def __init__(self, iteration_num, cost_value, loss_value, gradient_value, theta):
        self.iteration_num = iteration_num
        self.cost_value = cost_value
        self.loss_value = loss_value
        self.gradient_value = gradient_value
        self.theta = theta


class GradientDescentResult:
    """
    Container to contain result of gradient descent execution
    """

    def __init__(self):
        self.snapshots = []
        self.thetas = None

    def has_snapshots(self):
        return len(self.snapshots) > 0

    def save_result(self, thetas):
        self.thetas = thetas

    def save_snapshot(self, i, cost, loss, gradient, theta):
        self.snapshots = self.snapshots.append(GradientDescentSnapshot(i, cost, loss, gradient, theta))


class GradientDescentConfiguration:
    """
    Configuration to run gradient descent
    """

    def __init__(self, iterations_count, loss_func, hypothesis_func, learning_rate, lambda_value, is_save_snapshots):
        self.iterations_count = iterations_count
        self.loss_func = loss_func
        self.hypothesis_func = hypothesis_func
        self.learning_rate = learning_rate
        self.lambda_value = lambda_value
        self.is_save_snapshots = is_save_snapshots

    def iterations_count(self):
        return self.iterations_count

    def loss_func(self):
        return self.loss_func

    def hypothesis_func(self):
        return self.hypothesis_func

    def learning_rate(self):
        return self.learning_rate

    def lambda_value(self):
        return self.lambda_value

    def is_save_snapshots(self):
        return self.is_save_snapshots

    def __str__(self):
        return str.format("Iterations count: %s\n Learning rate: %s\n Lambda value: %s\n Snapshot will be saving: %s\n",
                          self.iterations_count(), self.lambda_value(), self.is_save_snapshots())


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
# ones line could be inserted BEFORE
def gradient_descent(features, results, gd_config):
    """
    Core of gradient descent execution
    Gradient descent realization that accepts features, results (Xs and Ys) and configuration for gradient descent
    :param features: Xs
    :param results: Ys
    :param gd_config: configuration to run gradient descent
    :return: packed gradient descent result, which has configured thetas and can has snapshots of execution
    """
    logging.info("Running gradient descent with next params: %s", gd_config)
    m = len(features)
    thetas = np.ones((features.shape[1], 1))
    gd_result = GradientDescentResult()

    for i in range(gd_config.iterations_count()):
        h = gd_config.hypothesis_func()(features, thetas)
        loss = gd_config.loss_func()(h, features)
        # why thetas starting from 1?
        lambda_component = (gd_config.lambda_value() / (2 * m)) * np.sum(thetas[1:] ** 2)
        cost = (np.sum(loss ** 2) / m) + lambda_component
        gradient = np.zeros(thetas.shape)
        gradient[0] = ((features.T @ (h - results))[0] / m)
        gradient[1:] = ((features.T @ (h - results)) / m)[1:] + ((gd_config.lambda_value() * thetas[1:]) / m)
        thetas = thetas - gd_config.learning_rate() * gradient
        if gd_config.is_save_snapshots():
            gd_result.save_snapshot(i=i, cost=cost, loss=loss, gradient=gradient, theta=thetas)

    gd_result.save_result(thetas=thetas)
    return gd_result
