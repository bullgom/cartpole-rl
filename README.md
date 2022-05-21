OpenAI/gym/CartPole-v1 게임을 RL/DQN을 사용해서 학습하는 레포지토리 입니다.

### 환경 구성
`conda env update -n my_env --file environment.yaml`

### 실행 환경
Windows 11, Python 3.9.12

### 실행 파일
`python train_off_policy.py`

### train_off_policy.py
hyper-parameter 지정, env-agent interface, training loop

### off_policy_agent.py 
Estimation을 위한 신경망 구조 정의, 행동 선택

### off_policy_teacher.py
loss 계산, batch backpropgation

### experience.py
experience replay buffer 정의

## 문제
### train_off_policy.py
간헐적으로 agent가 잘 합니다. 그런데 점수의 이동 평균(100)은 학습이 진행될수록 내려갑니다. 아래는 Reward Per Episode의 그래프 입니다.
![Rewards Per Episode](https://github.com/bullgom/cartpole-rl/blob/main/rewards_per_episode.png?raw=true)

### 이해가 안되는 차이점
1 - Wandb의 경우
그리고 400k일 때 500점에 도달했다.
https://wandb.ai/safijari/dqn-tutorial/reports/Deep-Q-Networks-with-the-Cartpole-Environment--Vmlldzo4MDc2MQ

참고할점 - Hyper Parameters - lr: 0.0001, Adam, batchsize: 100, min_epsilon: 0.05, steps_before_training: 10, epochs per target: 5000

학습을 굉장히 느리게 하도록 했다!

2 - 어느 Medium 글의 경우
https://medium.com/analytics-vidhya/solving-open-ais-cartpole-using-reinforcement-learning-part-2-73848cbda4f1

bs: 64, mem_size: 10000, target_up: 5, discount: 0.00, lr: 0.00025, discount gradient: 0.95, min_e: 0.001, RMSProp

아! 여기서는 이미지를 사용하지 않았다.
200 episode만에 max점수인 200점 도달했다
