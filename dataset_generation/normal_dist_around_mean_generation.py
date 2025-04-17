import random
from typing import List
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATAPOINTS_PER_DAY = 1
DATAPOINTS_PER_YEAR = DATAPOINTS_PER_DAY * 365

SIZE = DATAPOINTS_PER_YEAR


def translate(y: int, mean: int, std_dev: int) -> int:
    return y * std_dev + mean


def generate_days(mean: int, std_dev: int) -> List[int]:
    random_numbers_np_array = np.random.normal(loc=0, scale=1, size=SIZE)

    return np.vectorize(lambda x: round(translate(x, mean, std_dev)))(
        random_numbers_np_array
    )


print("Generating trainings data around random means")
xFileWrite = []
yFileWrite = []
for i in range(10_000):
    datapoints = generate_days(random.randint(7500, 15000), random.randint(1500, 3500))
    yNextList = []

    xFileWrite += [datapoints] * (SIZE + 1)
    for idx, datapoint in enumerate(datapoints):
        yFileWrite.append([f"The value at position {idx} is {datapoint}"])

    yFileWrite.append([f"The mean is {round(np.mean(datapoints))}"])


npX = np.array(xFileWrite)
npY = np.array(yFileWrite)

print(f"Writing to datasetX.npy, shape: {npX.shape}")
np.save("datasetX.npy", npX)

print(f"Writing to datasetY.npy, shape: {npY.shape}")
np.save("datasetY.npy", npY)


print("Sample X entries\n", npX[-5 * (SIZE + 1) :: (SIZE + 1)])
print("Sample Y entries\n", npY[-5 * (SIZE + 1) :: (SIZE + 1)])
