from snake_2 import SnakeEnv
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time

WIDTH, HEIGHT = 10, 10
snake = SnakeEnv([0, 0], 0, width=WIDTH, height=HEIGHT)
num_steps = 10 ** 6
FPS = 60

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
        epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000


def create_q_model():
    # Network defined by the DeepMind paper
    inputs = layers.Input(shape=(WIDTH, HEIGHT, 2))

    # Convolutions on the frames on the screen
    if WIDTH < 40:
        layer1 = layers.Conv2D(WIDTH - 2, 4, strides=2, activation="relu")(inputs)
        layer1a = layers.Conv2D(WIDTH - 2, 3, strides=1, activation="relu")(layer1)
        # layer1 = layers.Dense(8, activation="relu")(inputs)
        layer2 = layers.Flatten()(layer1a)
        layer3 = layers.Dense(32, activation="relu")(layer2)
        layer4 = layers.Dense(16, activation="relu")(layer3)
        sel_action = layers.Dense(4, activation="linear")(layer4)
    else:
        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 4, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 3, strides=2, activation="relu")(layer1)
        layer4 = layers.Flatten()(layer2)
        layer5 = layers.Dense(256, activation="relu")(layer4)
        sel_action = layers.Dense(4, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=sel_action)


model = create_q_model()
model_target = create_q_model()


optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 10000000
# Maximum replay length
# Note: The DeepMind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

t = time.time()
while True:
    state = np.array(snake.reset())
    # snake.render(1 / FPS)
    episode_reward = 0
    for _ in range(max_steps_per_episode):
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(4)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probability = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probability[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        state_next, reward, done = snake.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, 4)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}, time: {}"
            print(template.format(running_reward, episode_count, frame_count, time.time() - t))
            t = time.time()

            # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break
        # snake.render(1 / FPS)

    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 40:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
