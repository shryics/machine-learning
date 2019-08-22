import random
import collections
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# データの読込
iris = load_iris()
X, Y = iris.data, iris.target

# 特徴量の選択
feature_num = 2
features = X[:,feature_num]

best_thd, best_gini = None, 1
for i in range(1000):
    # 乱数による閾値の選択
    thd = random.choice(features)
    thd_list = list()

    # 分類
    for x, y in zip(features, Y):
        if x < thd:
            thd_list.append(y)

    # 不純度による評価
    try:
        # 分割前の不純度の計算
        N = len(Y)
        b_dic = collections.Counter(Y)
        geni_before = 1 - (b_dic[0] / N)**2 - (b_dic[1] / N)**2 - (b_dic[2] / N)**2
        # 分割後のルールを満たす不純度の計算
        M1 = len(thd_list)
        a_l_dic = collections.Counter(thd_list)
        gini_left = 1 - (a_l_dic[0] / M1)**2 - (a_l_dic[1] / M1)**2 - (a_l_dic[2] / M1)**2
        # 分割後のルールを満たさない不純度の計算
        M2 = N - M1
        a_r_dic = b_dic - a_l_dic
        gini_right = 1 - (a_r_dic[0] / M2)**2 - (a_r_dic[1] / M2)**2 - (a_r_dic[2] / M2)**2
        # 不純度の差分の計算
        gini = geni_before - gini_left * (M1 / N) - gini_right * (M2 / N)

        if gini < best_gini:
            best_gini = gini
            best_thd = thd

        plt.scatter(thd, gini)
    except:
        pass
plt.show()

print("best threshold:", best_thd)
print("best gini:", best_gini)