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
간헐적으로 agent가 잘 합니다. 그런데 점수의 이동 평균(100)은 학습이 진행될수록 내려갑니다.
