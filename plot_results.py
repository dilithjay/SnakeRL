import pickle
import matplotlib.pyplot as plt

algo = 'dqn_priority'
filename = 'results/' + algo + '_results.pickle'
with open(filename, "rb") as fp:
    score_list, max_score_list, running_reward_list, num_episodes_list = pickle.load(fp)

print("Score list:", score_list)
print("Max Score list:", max_score_list)
print("Reward list:", running_reward_list)
print("Num episodes list:", num_episodes_list)


plt.plot(range(1, len(score_list) + 1), running_reward_list)
plt.ylabel("Running Reward")
plt.xlabel("# of frames (in 10,000s)")
plt.show()

plt.plot(range(1, len(score_list) + 1), num_episodes_list)
plt.ylabel("# of episodes")
plt.xlabel("# of frames (in 10,000s)")
plt.show()

plt.plot(range(1, len(score_list) + 1), score_list)
plt.ylabel("Average Score")
plt.xlabel("# of frames (in 10,000s)")
plt.show()

plt.plot(range(1, len(score_list) + 1), max_score_list)
plt.ylabel("Max Score (Over 10 episodes)")
plt.xlabel("# of frames (in 10,000s)")
plt.show()
