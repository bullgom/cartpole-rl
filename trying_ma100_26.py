import gym
from agent import MyLovelyAgent, EpsilonGreedyParameters
from on_policy_teacher import MrAdamsTheTeacher
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


def make_reward_tensor(reward_sclar: float, action: int) -> torch.Tensor:
    zero = torch.zeros(2)
    zero[action] = reward_sclar
    return zero


if __name__ == "__main__":

    total_steps = 100000
    e_greedy_parameters = EpsilonGreedyParameters(0.07)
    discount = 0.99
    lr = 0.001
    buffer_size = 1000
    replay_per_step = 32
    image_size = 84

    display_game = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make('CartPole-v1')

    action_set = set([x for x in range(env.action_space.n)])

    agent = MyLovelyAgent(image_size, action_set,
                          e_greedy_parameters, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    buffer = ExperienceBuffer(buffer_size)
    teacher = MrAdamsTheTeacher(
        agent, discount, optimizer, device, buffer, replay_per_step)

    accumulated_reward = 0
    inner_step = 0
    episodes = 0
    episode_reward_ma = []
    ma_length = 100

    env.reset()
    image = make_state_image(env, False)
    proc_image = agent.transforms(image)
    loop_start = datetime.now()
    for t in range(total_steps):
        inner_step += 1

        if display_game:
            env.render("human")

        tensor_action = agent.make_up_my_mind(proc_image.unsqueeze(0))
        action = torch_scalar_to_int(tensor_action)

        _, reward, done, info = env.step(action)
        accumulated_reward += reward
        new_image = make_state_image(env, False)
        new_proc_image = agent.transforms(new_image)

        buffer.append(Experience(
            proc_image, new_proc_image, reward, action, done))
        image = new_image

        teacher.teach_multiple()

        if done:

            episode_reward_ma.append(accumulated_reward)
            if len(episode_reward_ma) > ma_length:
                episode_reward_ma.pop(0)
            ma = sum(episode_reward_ma)/len(episode_reward_ma)
            episodes += 1
            print(
                f"{episodes} - {t+1} - Steps Took: {inner_step} | AccumR: {accumulated_reward} | MA{ma_length}: {ma:.2f}")
            env.reset()
            accumulated_reward = 0.
            inner_step = 0

    env.close()
    loop_end = datetime.now()

    elapsed_time = (loop_end - loop_start).total_seconds()
    print(f"Elapsed Time: {elapsed_time} | Reward MA{ma_length}: {ma}")
