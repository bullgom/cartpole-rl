import gym
from agent import MyLovelyAgent, EpsilonGreedyParameters
from teacher import MrAdamsTheTeacher
import torch
import torchvision.transforms.functional as vision_f
from datetime import datetime


def torch_scalar_to_int(tensor: torch.Tensor) -> int:
    if not issubclass(type(tensor), torch.Tensor):
        raise TypeError(f"{type(tensor)} is not a Tensor")

    return tensor.item()


def make_state_image(env: gym.Env) -> torch.Tensor:

    image = env.render("rgb_array")
    image_tensor = vision_f.to_tensor(image).unsqueeze(0)
    return image_tensor

def make_reward_tensor(reward_sclar: float, action: int) -> torch.Tensor:
    zero = torch.zeros(2)
    zero[action] = reward_sclar
    return zero

if __name__ == "__main__":

    total_steps = 100000
    e_greedy_parameters = EpsilonGreedyParameters(0.05)
    discount = 0.99
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    display_game = False

    env = gym.make('CartPole-v1')

    action_set = set([x for x in range(env.action_space.n)])

    agent = MyLovelyAgent((600, 400, 3), action_set,
                          e_greedy_parameters, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    teacher = MrAdamsTheTeacher(agent, discount, optimizer, device)

    accumulated_reward = 0
    inner_step = 0
    episodes = 0
    episode_reward_ma = []
    ma_length = 30

    env.reset()
    image = make_state_image(env)
    loop_start = datetime.now()
    for t in range(total_steps):
        inner_step += 1

        if display_game:
            env.render("human")

        tensor_action = agent.make_up_my_mind(image)
        action = torch_scalar_to_int(tensor_action)

        _, reward, done, info = env.step(action)
        accumulated_reward += reward
        new_image = make_state_image(env)
        
        r_tensor = make_reward_tensor(reward, action).to(device)
        teacher.teach_a_problem(r_tensor, new_image, image)
        image = new_image

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
