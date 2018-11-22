import gym
import multiprocessing

lock = multiprocessing.Lock()
env = gym.make('meta-SuperMarioBros-Tiles-v0')
env.no_render = True
observation = env.reset()

done = False
alive = True
t = 0
while not done and alive:
	# print(observation)
	
	action = [1,0,0,1,1,0]  # choose random action #[, left, , right, up, ]
	observation, reward, done, alive, info = env.step(action)  # feedback from environment
	
	# print(info['life'])
	# if(info['life'] == 0):
	# 	print("mario dead")
		# break
	# print(done)
	# print(type(observation))

	t += 1	
	if not t % 1000:
		print(t, info)
env.close()