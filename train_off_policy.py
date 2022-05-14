import gym
from agent import MyLovelyAgent, EpsilonGreedyParameters
from off_policy_teacher import MrAdamsTheTeacher
import torch
import torchvision.transforms.functional as vision_f
from datetime import datetime
from experience import ExperienceBuffer, Experience


def torch_scalar_to_int(tensor: torch.Tensor) -> int:
    if not issubclass(type(tensor), torch.Tensor):
        raise TypeError(f"{type(tensor)} is not a Tensor")

    return tensor.item()


def make_state_image(env: gym.Env, inc_dim=True) -> torch.Tensor:

    image = env.render("rgb_array")

    image_tensor = vision_f.to_tensor(image)
    if inc_dim:
        image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


if __name__ == "__main__":

    total_steps = 10000
    e_greedy_parameters = EpsilonGreedyParameters(0.01)
    discount = 0.99
    lr = 0.01
    buffer_size = 1000
    replay_per_step = 128
    image_size = 84
    episodes_per_copy = 10

    display_game = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make('CartPole-v1')

    action_set = set([x for x in range(env.action_space.n)])

    policy_agent = MyLovelyAgent(image_size, action_set,
                          e_greedy_parameters, device)
    target_agent = policy_agent.copy()
    target_agent.eval()
    
    optimizer = torch.optim.Adam(policy_agent.parameters(), lr=lr)
    buffer = ExperienceBuffer(buffer_size)
    teacher = MrAdamsTheTeacher(target_agent, policy_agent, discount, 
                                optimizer, device, buffer, replay_per_step)

    accumulated_reward = 0
    inner_step = 0
    episodes = 0
    episode_reward_ma = []
    total_reward = []
    max_ma_reward = 0
    ma_length = 100

    env.reset()
    image = make_state_image(env, False)
    proc_image = policy_agent.transforms(image)
    loop_start = datetime.now()
    for t in range(total_steps):
        inner_step += 1

        if display_game:
            env.render("human")

        tensor_action = policy_agent.make_up_my_mind(proc_image.unsqueeze(0))
        action = torch_scalar_to_int(tensor_action)

        _, reward, done, info = env.step(action)
        accumulated_reward += reward
        new_image = make_state_image(env, False)
        new_proc_image = policy_agent.transforms(new_image)

        buffer.append(Experience(
            proc_image, new_proc_image, reward, action, done))
        image = new_image

        teacher.teach_multiple()
        

        if done:
            total_reward.append(accumulated_reward)
            episode_reward_ma.append(accumulated_reward)
            if len(episode_reward_ma) > ma_length:
                episode_reward_ma.pop(0)
            ma = sum(episode_reward_ma)/len(episode_reward_ma)
            if ma > max_ma_reward:
                max_ma_reward = ma

            if episodes % episodes_per_copy == 0:
                print("Copying value function")
                teacher.target.load_state_dict(policy_agent.state_dict())
                
            episodes += 1
            print(
                f"{episodes} - {t+1} - Steps Took: {inner_step} | AccumR: {accumulated_reward} | MA{ma_length}: {ma:.2f}")
            env.reset()
            accumulated_reward = 0.
            inner_step = 0

    env.close()
    loop_end = datetime.now()

    elapsed_time = (loop_end - loop_start).total_seconds()
    print(
        f"Elapsed Time: {elapsed_time} | Max Reward MA{ma_length}: {max_ma_reward}")

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import matplotlib.pyplot as plt
    plt.plot(total_reward)
    plt.show()
