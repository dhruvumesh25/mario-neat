"""
mario - solver
"""

from __future__ import print_function
import os
import neat
import visualize
import gym
import multiprocessing
import time
import subprocess
from multiprocessing import Process, Value, Array

def killFCEUX():
    bashCommand = "pkill -9 fceux"
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def binarize(inp):
    out = []
    for i in inp:
        if i < 0:
            out.append(0)
        else:
            out.append(1)
    return out

def eval_single_genome(genome, genome_id, index, config, return_dict, write_lock):
    # print("inside process", genome_id)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # lock = multiprocessing.Lock()
    env = gym.make('meta-SuperMarioBros-Tiles-v0')
    # env.lock = lock
    # env.no_render = True
    # env.lock.acquire()
    observation = env.reset()
    env.locked_levels = [False] * 32
    env.change_level(new_level=0)
    # env.lock.release()

    done, alive = False, True
    info, reward = None, None
    time_stale = 0
    max_dist = 0

    # ctr = 0

    while not done and alive and time_stale < 80:
        # print("in loop",genome_id)
        # print(observation)
        action = net.activate(observation.flatten().tolist())
        bin_action = []
        for i in action:
            if i < 0:
                bin_action.append(0)
            else:
                bin_action.append(1)

        observation, reward, done, alive, info = env.step(bin_action)  # feedback from environment
        if 'ignore' in info:
            # print('ignore in',genome_id)
            # done = False
            # env = gym.make('meta-SuperMarioBros-Tiles-v0')
            # # env.lock.acquire()
            # env.reset()
            # env.locked_levels = [False] * 32
            # env.change_level(new_level=0)
            # # env.lock.release()
            # time_stale = 0
            # prev_dist = 0

            break

        if info['distance'] <= max_dist:
            time_stale += 1
        else:
            time_stale = 0
            max_dist = info['distance']

        # if not ctr%1000:
        #     print("inside loop", genome_id)
        # ctr+=1;

    # print("outside loop", genome_id)

    # env.lock.acquire()
    # env.close()
    # env.lock.release()

    write_lock.acquire()
    return_dict[genome_id] = max_dist
    write_lock.release()

    print("evaluating genome", genome_id, "fitness=", return_dict[genome_id])

def eval_genomes(genomes, config):
    counter = 0
    num_genomes = len(genomes)

    p = [None] * num_genomes

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    max_parallel_processes = 5

    lock = multiprocessing.Lock()
    for i in range(0,num_genomes,max_parallel_processes):
        for j in range(max_parallel_processes):
            if i+j >= num_genomes:
                continue
            else:
                genome_id, genome = genomes[i+j]
                p[i+j] = Process(target=eval_single_genome, args=(genome,genome_id,i+j,config,return_dict,lock,))
                p[i+j].start()
        for j in range(max_parallel_processes):
            if i+j >= num_genomes:
                continue
            else:
                p[i+j].join()
                pass
        killFCEUX()

    print(return_dict)

    for genome_id, genome in genomes:
        genome.fitness = return_dict[genome_id]

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(300))

    # Run for up to 3000 generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)