import gym, os
import multiprocessing

# def get_pid(name):
#     return check_output(["pidof", name]).split()

# def killFCEUX():
#     for pid in get_pid("fceux"):
#         pid = int(pid)
#         os.kill(pid, signal.SIGKILL)

lock = multiprocessing.Lock()
envName = 'meta-SuperMarioBros-Tiles-v0'

env = gym.make(envName)
env.lock = lock
env.lock.acquire()
env.reset()
env.lock.release()
results = []
env.locked_levels = [False] * 32

maxDistance = 0
distance = None
staleness = 0
scores = []
finalScore = 0
done = False
maxReward = 0
for LVint in range(1):
    maxDistance = 0
    oldDistance = 0
    bonus = 0
    bonusOffset = 0
    staleness = 0
    done = False
    env.change_level(new_level=LVint)
    while not done:
        ob = env.tiles.flatten()
        action = [1,0,0,1,0,0]
        ob, reward, done, _ = env.step(action)
        if 'ignore' in _:
            done = False
            env = gym.make('meta-SuperMarioBros-Tiles-v0')
            env.lock.acquire()
            env.reset()
            env.locked_levels = [False] * 32
            env.change_level(new_level=LVint)
            env.lock.release()
        distance = env._get_info()["distance"]
        if oldDistance - distance < -100:
            bonus = maxDistance
            bonusOffset = distance
        if maxDistance - distance > 50 and distance != 0:
            maxDistance = distance
        if distance > maxDistance:
            maxDistance = distance
            staleness = 0
        if maxDistance >= distance:
            staleness += 1

        if staleness > 80 or done:
            scores.append(maxDistance - bonusOffset + bonus)
            if not done:
                print("heyyy")
                done = True
        oldDistance = distance
for score in scores:
    finalScore += score
finalScore = round(finalScore / 32)
results.append(finalScore)

# killFCEUX()
print(results)