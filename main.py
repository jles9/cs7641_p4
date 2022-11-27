import random
import numpy as np

import policy_iter
import value_iter
import qlearn


def main():
    random.seed(42)
    np.random.seed(42)

    policy_iter.run_PI_exp()
    value_iter.run_VI_exp()
    qlearn.run_QLearn_exp()



if __name__ == "__main__":
    main()