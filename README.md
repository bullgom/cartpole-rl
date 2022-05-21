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

Wandb의 기능을 확인하니 Hyper Paremter 공간을 알아서 탐색해주는 기능이 있는 것으로 보였다. 

사실 기존 실험들의 실패 요인을 아직도 정확히 모르겠다. 하나 하나 Pytorch와 다른 코드들의 코드로 변경하다 보니 어느새 되었다. 그리고 또하나, 대부분의 실험을 400 episode 수준에서 종료 시켰다. 어쩌다가 그보다 더 실행해 보았는데 후반에서야 조금씩 학습하기 시작했다.


해당 글에서 다음과 같은 문장이 있었다: "Hyperparameters are a particular pain point in reinforcement learning, much more so than in deep learning since it can take a long time before any signs of progress show up."

코드의 문제인지, Hyper Parameter 문제인지 아니면 아직 학습이 덜 된 것인지 모르겠다. 


2 - 어느 Medium 글의 경우
https://medium.com/analytics-vidhya/solving-open-ais-cartpole-using-reinforcement-learning-part-2-73848cbda4f1

bs: 64, mem_size: 10000, target_up: 5, discount: 0.00, lr: 0.00025, discount gradient: 0.95, min_e: 0.001, RMSProp

아! 여기서는 이미지를 사용하지 않았다.
200 episode만에 max점수인 200점 도달했다

## To Do
이제 되긴 되는데, 어느 부분이 문제였는지 정확히 모르겠다.

일단 Hyper Parameter는 확실히 문제였다. Wandb 글에서 lr 이 크면 부정적인 영향을 미친다고 나와 있었다. 나는 0.01 수준으로 나름 큰 lr을 쓰고 있었기 때문에, 이게 문제였을 수 있다.

이제 나의 코드로 다시 해 볼 차례이다.
