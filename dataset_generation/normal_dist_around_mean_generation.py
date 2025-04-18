import random
from typing import List
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATAPOINTS_PER_DAY = 1
DATAPOINTS_PER_YEAR = DATAPOINTS_PER_DAY * 365

# SIZE = DATAPOINTS_PER_YEAR
SIZE = 20


def generate_test_data(included_indices=[]):
    def translate(y: int, mean: int, std_dev: int) -> int:
        return y * std_dev + mean

    def generate_days(mean: int, std_dev: int) -> List[int]:
        # random_numbers_np_array = np.random.normal(loc=0, scale=1, size=SIZE)
        random_numbers_np_array = np.random.normal(loc=0, scale=1, size=SIZE)

        return np.vectorize(lambda x: round(translate(x, mean, std_dev)))(
            random_numbers_np_array
        )

    print("Generating trainings data around random means")
    xFileWrite = []
    yFileWrite = []
    for _ in range(2**11):
        datapoints = generate_days(
            random.randint(7500, 15000), random.randint(1500, 3500)
        )
        # yNextList = []

        xFileWrite += [datapoints] * len(included_indices)
        # xFileWrite += [datapoints] * (SIZE + 1)
        for idx, datapoint in enumerate(datapoints):
            if idx not in included_indices:
                continue
            yFileWrite.append(f"{datapoint}")

        # yFileWrite.append([f"The mean is {round(np.mean(datapoints))}"])

    npX = np.array(xFileWrite)
    npY = np.array(yFileWrite)

    print(f"Writing to datasetX.npy, shape: {npX.shape}")
    # np.save("datasetX.npy", npX)

    print(f"Writing to datasetY.npy, shape: {npY.shape}")
    # np.save("datasetY.npy", npY)

    # print("Sample X entries\n", npX[-5 * (SIZE + 1) :: (SIZE + 1)])
    # print("Sample Y entries\n", npY[-5 * (SIZE + 1) :: (SIZE + 1)])
    print(
        "Sample X entries\n",
        npX[-5 * (len(included_indices)) :: (len(included_indices))],
    )
    print(
        "Sample Y entries\n",
        npY[-5 * (len(included_indices)) :: (len(included_indices))],
    )
    return npX, npY
