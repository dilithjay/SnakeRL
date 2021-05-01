import pickle
import matplotlib.pyplot as plt

with open("dqn_results.pkl", "rb") as fp:
    score_list, max_score_list, running_reward_list, num_episodes_list = pickle.load(fp)

print("Score list:", score_list)
print("Max Score list:", max_score_list)
print("Reward list:", running_reward_list)
print("Num episodes list:", num_episodes_list)


plt.plot(range(1, len(score_list) + 1), running_reward_list)
plt.show()

plt.plot(range(1, len(score_list) + 1), num_episodes_list)
plt.show()

"""plt.plot(range(1, len(score_list) + 1), score_list)
plt.show()

plt.plot(range(1, len(score_list) + 1), max_score_list)
plt.show()"""
