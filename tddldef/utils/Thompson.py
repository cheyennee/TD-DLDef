import numpy as np
import random

class ThompsonSampling:
    def __init__(self, layers, alpha, beta):
        """
        初始化汤普森采样算法
        :param n_arms: 臂的数量
        """
        self.__layers = layers

        # 初始化每个臂的成功次数和失败次数
        self.__dict = {}
        for layer in layers:
            self.__dict[layer] = [alpha, beta] # 成功次数 & 失败次数
        self.__pre = ""
        self.__cnt = 0

        self.record = {} # 字典记录 选择的结果，包括 <A,B> 和 A 类型的结果，用来判断 reward
        self.__single_record = set()

    def choose_arm(self, candidates):
        """
        选择臂：从每个臂的后验分布中采样，选择值最大的臂
        """
        successes = np.zeros(len(candidates))
        failures = np.zeros(len(candidates))
        for i, candidate in enumerate(candidates):
            successes[i] = self.__dict[candidate][0]
            failures[i] = self.__dict[candidate][1]


        # 从每个臂的Beta分布中采样
        sampled_values = np.random.beta(successes + 1, failures + 1)
        # 选择采样值最大的臂
        re = np.argmax(sampled_values)
        name = candidates[int(re)]

        return name

    def update(self, chosen_name, reward):
        """
        更新所选臂的成功和失败次数
        :param chosen_name: 选择的臂
        :param reward: 奖励（0 或 1）
        """
        self.__single_record.add(chosen_name)
        if self.record.get(chosen_name, None) is None:
            self.record[chosen_name] = 1
        else:
            self.record[chosen_name] += 1

        if self.record.get(self.__pre+chosen_name, None) is None:
            self.record[self.__pre+chosen_name] = 1
        else:
            self.record[self.__pre+chosen_name] += 1
        self.__pre = chosen_name
        self.__cnt += 1

        if reward == 1:
            self.__dict[chosen_name][0] += 1
        else:
            self.__dict[chosen_name][1] += 1


    def is_reward(self, selected_name):

        selected_cnt = self.record.get(selected_name, None)
        if selected_cnt is None:
            return 1
        selected_cnt2 = self.record.get(self.__pre+selected_name, None)
        if selected_cnt2 is None:
            return 1

        if selected_cnt / self.__cnt < 1 / len(self.__layers):
            return 1
        return 0

    def print(self):

        for layer in self.__layers:
            print(f'{layer} 次数 {self.record.get(layer, None)}')
        print(self.record)

    def cal_coverage(self):
        return round(100 * len(self.__single_record) / len(self.__layers), 2), self.__single_record

if __name__ == '__main__':
    layers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    a_sampling = ThompsonSampling(layers=layers, alpha=1, beta=1)

    for i in range(1000):
        cands = random.sample(layers, 5)
        re = a_sampling.choose_arm(cands)
        print(f'选择的结果 {re}')
        a_sampling.update(re, a_sampling.is_reward(re))
    a_sampling.print()
