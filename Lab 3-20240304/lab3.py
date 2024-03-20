# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
import numpy as np
x_train = np.array([1.0, 2.0])
y_train = np.array([300.5, 500.0])
# print(f"x_train = {x_train}")
# print(f"y_train = {y_train}")
# print(f"x_train.shape: {x_train.shape}")
# m = x_train.shape[0]
# print(f"Number of training examples is: {m}")
# m = len(x_train)
# print(f"Number of training examples is: {m}")


# w = 200                         
# b = 100    
# x_i = 1.2
# cost_1200sqft = w * x_i + b    

# print(f"${cost_1200sqft:.0f} thousand dollars")

i = 0 # Change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# plt.scatter(x_train, y_train, marker='x', c='r')
# # Set the title
# plt.title("Housing Prices")
# # Set the y-axis label
# plt.ylabel('Price (in 1000s of dollars)')
# # Set the x-axis label
# plt.xlabel('Size (1000 sqft)')
# plt.show()

# w = 100
# b = 100
# print(f"w: {w}")
# print(f"b: {b}")

