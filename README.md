OpenAI/gym/CartPole-v1 게임을 RL/DQN을 사용해서 학습하는 레포지토리 입니다.

마지막에 참지 못하고 PyTorch에서 Tutorial을 봐버렸습니다.
해당 튜토리얼 중 따라한 내용은 
  - 아키텍쳐
  - Policy를 학습하고, Target에 복사한다는 점 (보기 전에 Target을 업데이트, Policy에 복사 했습니다.)
  - 파라미터
    - ER size
    - episodes per copy
    - batch size (replay_per_step)

이외 내용은 책과 수업 그리고 코드 이외의 자료 통해 작성했습니다.

다양한 시도들이 있었습니다.
 1. 맨 처음 Experience Replay와 On-Policy로 진행 했습니다. -> FAIL
 2. Experience Replay + On-Policy -> ma100 26 점
 3. Experience Replay + Off-Policy -> ma100 14 점

어려운 프로젝트였습니다.
 1. 문제의 요인들이 다양했습니다.
    - (파라미터, 코드 에러, 알고리즘 등)
 2. 같은 세팅에서도 여러가지 결과가 나옵니다.
    - trying_ma100_26.py을 실행했을 때 어쩔 때에는 아예 학습을 하지 못 합니다.
    - train_off_policy.py을 실행했을 때 최대 ma100가 14까지 나오다가 다시 12까지 내려가는 등...

#GG
    
