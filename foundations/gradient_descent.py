class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x_not = init

        def function(x):
            return x**2
        def function_hat(x):
            return 2*x

        if iterations == 0:
            return x_not
        else:
            for i in range(iterations):
                x_new = x_not - learning_rate * function_hat(x_not)
                x_not = x_new
            return round(x_new, 5)
        

solution = Solution()

iterations = 0
learning_rate = 0.01
init = 5

solution.get_minimizer(iterations, learning_rate, init)
