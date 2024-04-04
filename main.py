from Test import Test
import Utils
from Train import Train


if __name__ == '__main__':

    utils = Utils.Utils()
    train_obj = Train(utils=utils)
    train_obj.train_policy()
    test_obj = Test(utils=utils)
    test_obj.get_goal_directed_actions()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
