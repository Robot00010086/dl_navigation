
conda create -n deeprl_navigation python=3.6

conda activate deeprl_navigation

conda install pytorch cudatoolkit=9.0 -c pytorch

pip install -r requirements.txt



dqn_agent.py   Agent and ReplayBuffer

model.py    QNetwork

test.py   test if enviroment is available

train.py   dqn function(train a single agent in an env)

train_sessions.py  train several agents with diffrent parameters in a sequential manner

eval.py   test file if the model has been trained (mannual copy parameter from training output)

utils.py  read or wirte log file 

