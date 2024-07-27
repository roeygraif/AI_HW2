from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import numpy as np

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    package = robot.package
    charging_station = env.charge_stations[robot_id]
    bonus = 0
    dist_to_charging_station = manhattan_distance(robot.position, charging_station.position)
    charging_weight = 0
    if package:
        target_dist = manhattan_distance(robot.position, package.destination)
        bonus = 2*manhattan_distance(package.position, package.destination)
    else:
        target_dist = min([manhattan_distance(robot.position, package.position) for package in env.packages if package.on_board])


    if robot.battery <= dist_to_charging_station:
        charging_weight = 10 - dist_to_charging_station
    return 100-target_dist + 2000*robot.credit + 10*bonus + 1000 * charging_weight



class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        tree_height = 0
        child_ops, children_env = self.successors(env,agent_id)
        child_list = []
        while time.time() < start_time + time_limit - 0.7:
            child_list = [self.mini_max_RB(child_env, agent_id, tree_height, (agent_id+1)%2) for child_env in children_env]
            print(child_list)
            tree_height = tree_height + 1
        if child_list is []:
            return None
        return child_ops[child_list.index(max(child_list))]
    
    def mini_max_RB(self, env: WarehouseEnv, agent_id, tree_height, turn):
        if tree_height == 0 or env.done():
            return smart_heuristic(env, agent_id)
        child_ops, children_env = self.successors(env, agent_id)
        for child_env in children_env:
            if turn is agent_id:
                ret_val = -1 * np.inf
                val = self.mini_max_RB(child_env, agent_id, tree_height - 1, (turn+1)%2)
                ret_val = max(val, ret_val)
            else:
                ret_val = np.inf
                val = self.mini_max_RB(child_env, agent_id, tree_height - 1, (turn+1)%2)
                ret_val = min(val, ret_val)
        return ret_val

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)