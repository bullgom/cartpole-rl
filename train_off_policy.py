import matplotlib.pyplot as plt
import gym
from agent import MyLovelyAgent, EpsilonGreedyParameters
from off_policy_teacher import MrAdamsTheTeacher
import torch
import torchvision.transforms.functional as vision_f
from torchvision.transforms import transforms as tr
from datetime import datetime
from er import ReplayMemory, Transition

import numpy as np
from typing import Union
from PIL import Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot(rewards, mean_values):
    plt.clf()
    plt.title("Training!!!!")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.plot(rewards)
    plt.plot(mean_values)
    plt.pause(0.001)


def make_state_image(env: gym.Env, transforms) -> torch.Tensor:
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return transforms(screen).unsqueeze(0)


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


if __name__ == "__main__":
    total_steps = 20000
    e_greedy_parameters = EpsilonGreedyParameters(.9, 0.01, 200)
    discount = 0.999
    lr = 0.00025
    buffer_size = 10000
    replay_per_step = 128
    image_size = (40, 90)
    window_size = (600, 400)
    episodes_per_copy = 30

    display_game = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = tr.Compose([
        tr.ToPILImage(),
        tr.Grayscale(),
        tr.Resize(40, Image.CUBIC),
        tr.ToTensor()
    ])

    env = gym.make('CartPole-v1')

    action_set = set([x for x in range(env.action_space.n)])

    policy_agent = MyLovelyAgent(image_size, action_set,
                                 e_greedy_parameters, device)

    target_agent = policy_agent.copy()
    target_agent.eval()

    optimizer = torch.optim.Adam(policy_agent.parameters(), lr=lr)
    buffer = ReplayMemory(buffer_size)
    teacher = MrAdamsTheTeacher(target_agent, policy_agent, discount,
                                optimizer, device, buffer, replay_per_step)

    accumulated_reward = 0
    inner_step = 0
    episodes = 0
    episode_reward_ma = []
    ma_records = []
    total_reward = []
    episode_loss = []
    max_ma_reward = 0
    ma_length = 100

    plt.show()
    plot(total_reward, ma_records)
    env.reset()
    last_image = make_state_image(env, transforms)
    new_image = make_state_image(env, transforms)
    state = new_image - last_image

    loop_start = datetime.now()
    for t in range(total_steps):
        inner_step += 1

        if display_game:
            env.render("human")

        action = policy_agent.make_up_my_mind(state)

        _, reward, done, info = env.step(action.item())
        reward_tensor = torch.tensor([reward], device=device)
        accumulated_reward += reward
        
        last_image = new_image
        new_image = make_state_image(env, transforms)
        if not done:
            next_state = new_image - last_image
        else:
            next_state = None
        

        buffer.push(state, action, next_state, reward_tensor)
        state = next_state

        loss = teacher.teach_multiple()
        if loss:
            episode_loss.append(loss.item())

        if done:
            total_reward.append(accumulated_reward)
            episode_reward_ma.append(accumulated_reward)
            if len(episode_reward_ma) > ma_length:
                episode_reward_ma.pop(0)
            ma = sum(episode_reward_ma)/len(episode_reward_ma)
            ma_records.append(ma)
            if ma > max_ma_reward:
                max_ma_reward = ma

            if episodes % episodes_per_copy == 0:
                print("Copying value function")
                teacher.target.load_state_dict(policy_agent.state_dict())

            episodes += 1
            avg_loss = sum(episode_loss) / \
                len(episode_loss) if episode_loss else 0
            print(
                f"{episodes} - {t+1} - Steps Took: {inner_step} | AccumR: {accumulated_reward} | MA{ma_length}: {ma:.2f} | Loss: {avg_loss}")
            plot(total_reward, ma_records)
            env.reset()
            accumulated_reward = 0.
            inner_step = 0
            e_greedy_parameters.step()
            episode_loss.clear()
            
            last_image = make_state_image(env, transforms)
            new_image = make_state_image(env, transforms)
            state = new_image - last_image

    env.close()
    loop_end = datetime.now()

    elapsed_time = (loop_end - loop_start).total_seconds()
    print(
        f"Elapsed Time: {elapsed_time} | Max Reward MA{ma_length}: {max_ma_reward}")

    plt.plot(total_reward, label="Reward Per Episode")
    plt.plot(ma_records, label="MA 100")
    plt.legend()
    plt.show()
