from Test import Test
import Utils
from Train import Train


if __name__ == '__main__':

    utils = Utils.Utils()
    train_obj = Train(utils=utils)
    train_obj.train_policy()
    test_obj = Test(utils=utils)
    test_obj.test_random_goal_selection()
    # test_obj.test_agents_at_all_locations()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
