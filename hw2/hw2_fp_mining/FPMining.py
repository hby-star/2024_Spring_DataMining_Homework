from Apriori import Apriori
from FPGrowth import FPGrowth
from ECLAT import ECLAT
import time


# example
# input
# [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
# output
# [[1], [3], [2], [5], [1, 3], [2, 3], [3, 5], [2, 5], [2, 3, 5]]


def loadDataSet():
    dataset = []
    with open('./data/DBLPdata-10k.txt', 'r') as file:
        for line in file:
            dataset.append(line.strip().split(','))

    return dataset


def main():
    dataset = loadDataSet()
    min_sup = 5

    print(f"Dataset len: {len(dataset)}")
    print(f"Min support: {min_sup}")
    print("")

    print("Running Apriori ...")
    start_time = time.time()
    apriori_result = Apriori(dataset, min_sup)
    end_time = time.time()
    apriori_time = end_time - start_time
    print(f"Apriori time: {apriori_time}")
    print(f'Apriori result len: {len(apriori_result)}')
    print(apriori_result)
    print("")

    print("Running FPGrowth ...")
    start_time = time.time()
    fp_growth_result = FPGrowth(dataset, min_sup)
    end_time = time.time()
    fp_growth_time = end_time - start_time
    print(f"FPGrowth time: {fp_growth_time}")
    print(f'FPGrowth result len: {len(fp_growth_result)}')
    print(fp_growth_result)
    print("")

    print("Running ECLAT ...")
    start_time = time.time()
    eclat_result = ECLAT(dataset, min_sup)
    end_time = time.time()
    eclat_time = end_time - start_time
    print(f"ECLAT time: {eclat_time}")
    print(f'ECLAT result len: {len(eclat_result)}')
    print(eclat_result)
    print("")


if __name__ == '__main__':
    main()
