# DQN Scheduling algorithm 
 The agent’s goal is to maximize the total expected reward over all possible trajectories, even though we defined finite states and action space, there are still a huge number of trajectories, which motivates the use of reinforcement learning. It can be converted as an iterative update in the deep-Q network, which is proposed by Watkins as follows:    Q(S_t,A_t )=Q(S_t,A_t )+α[r_(t+1)+γMaxQ(S_(t+1),A_(t+1) )-Q(S_t,A_t )] 
 Where left Q(S_t,A_t ) is the updating Q-values (rewards) at state S_t execute action A_t. r_(t+1)+γMaxQ(S_(t+1),A_(t+1) ) is the predicted target-Q value, where r_(t+1) is the reward when executing action A_(t+1) from state A_t into state A_(t+1).a is learning rate. MaxQ(S_(t+1),A_(t+1) ) is maximum Q-value after executing all possible actions A_(t+1). In DQN, will adopt deep neural network for predicting the Q-values.

![image](https://user-images.githubusercontent.com/31919004/114265443-6afb3600-9a0e-11eb-9d61-e30abddbc1b5.png)








## problem description

In the process JSP, we give one example to explain MDP. Assume we have jobs J={J_1,J_2,J_3,⋯,J_J} need to be processed using M={m_1,m_2,m_3,⋯,m_M} machines, each job consists of M different sub-tasks and only can be processed by a different machine. To simplify the problem, there are some constraints: 
Constrain 1: Each sub-task only needs to be processed once at m machine. 
Constrain 2: Each sub-task has the time limitations T_j={T_j1,T_j2,⋯,T_jM}, especially, the lowest time we defined is ten minutes, the highest time is fifty minutes. 
Constrain 3: Each job has special orders O_j={J_j1,J_j2,⋯,J_jM} for M sub-tasks. 
Constrain 4: The process is done when finishing all jobs.
According to the above constraints, define processing order O={O_1,O_2,⋯,O_J} and time T={T_1,T_2,⋯,T_J} of JSP as shown in Table 1. The example shows five jobs that are processed by four machines, as shown in Table 2

![image](https://user-images.githubusercontent.com/31919004/114265428-5c148380-9a0e-11eb-83d3-7aa658372389.png)







```python 
self.M_processing_order = np.array(
    [[1, 3, 0, 2], [0, 2, 1, 3], [3, 1, 2, 0], [1, 3, 0, 2], [0, 1, 2, 3]])
self.M_processing_time = np.array([[18, 20, 21, 17], [18, 26, 15, 16], [
    17, 18, 27, 23], [18, 21, 25, 15], [22, 29, 28, 21]])
```



## Output of the code 

``` txt
loop : 840/10000,  score: 143.0 success: 0 / 10, e: 0.014
[3, 0, 1, 2, 4, 3, 1, 0, 2, 4, 0, 3, 2, 1, 4, 3, 0, 1, 2, 4] 20
loop : 850/10000,  score: 143.0 success: 1 / 10, e: 0.014
[1, 0, 3, 1, 2, 4, 0, 3, 2, 4, 0, 3, 1, 2, 1, 4, 0, 3, 2, 1] 20
loop : 860/10000,  score: 143.0 success: 0 / 10, e: 0.013
[3, 1, 0, 4, 3, 1, 0, 2, 4, 2, 0, 3, 2, 1, 4, 3, 1, 0, 4, 2] 20
loop : 870/10000,  score: 143.0 success: 1 / 10, e: 0.012
[4, 3, 0, 3, 1, 1, 0, 4, 2, 2, 4, 0, 3, 1, 4, 2, 3, 1, 0, 4] 20
loop : 880/10000,  score: 143.0 success: 1 / 10, e: 0.012
[1, 0, 4, 3, 2, 1, 0, 4, 2, 3, 0, 3, 1, 4, 2, 1, 0, 4, 3, 0] 20
```

# The random 4*5 job shop problem 

## problem description

random 4*5 problem create by function 

JobShop.py line 100


## Output of the code 

``` txt

[2, 1, 0, 3, 4, 1, 0, 3, 2, 4, 2, 0, 3, 1, 4, 2, 0, 1, 3, 4] 20
loop : 2720/10000,  score: 156.0 success: 7 / 10, e: 0.01
[0, 3, 1, 4, 2, 1, 4, 0, 3, 2, 4, 0, 3, 2, 1, 0, 3, 1, 4, 2] 20
loop : 2730/10000,  score: 172.0 success: 3 / 10, e: 0.01
[2, 2, 1, 4, 3, 0, 3, 1, 4, 2, 0, 0, 3, 1, 4, 2, 1, 4, 3, 0] 20
loop : 2740/10000,  score: 143.0 success: 2 / 10, e: 0.01
[2, 1, 0, 3, 1, 4, 0, 3, 2, 4, 2, 0, 3, 1, 4, 2, 1, 0, 3, 4] 20
loop : 2750/10000,  score: 143.0 success: 2 / 10, e: 0.01
[0, 1, 3, 2, 1, 4, 0, 3, 2, 4, 0, 3, 2, 1, 4, 0, 1, 3, 2, 4] 20
loop : 2760/10000,  score: 143.0 success: 4 / 10, e: 0.01
[2, 1, 0, 3, 1, 0, 4, 2, 3, 4, 2, 0, 1, 3, 4, 2, 1, 0, 3, 4] 20

```
