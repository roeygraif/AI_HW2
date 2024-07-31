from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut

# TODO: section a : 3
#def charge(env: WarehouseEnv, robot_id: int, dist_to_charging_station, target_dist):
#    robot = env.get_robot(robot_id)
#    rival_robot = env.get_robot((robot_id+1)%2)
#    if rival_robot.credit > robot.credit and dist_to_charging_station < robot.battery and dist_to_charging_station > robot.battery-3 and robot.credit>0:
#        if robot.package:
#            return True
#        else:
#            package = [package for package in env.packages if manhattan_distance(package.position, robot.position)==target_dist and package.on_board][0]
#            if target_dist + manhattan_distance(package.position, package.destination) > robot.battery and robot.credit>0:
#                return True
#    return False
        

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    rival_robot = env.get_robot((robot_id+1)%2)
    package = robot.package
    bonus = 0
    losing_weight = 0
    if package:
        target_dist = manhattan_distance(robot.position, package.destination)
        bonus = 2*manhattan_distance(package.position, package.destination)
    else:
        target_dist = min([manhattan_distance(robot.position, package.position) for package in env.packages if package.on_board])

    return 20-target_dist + 40*robot.credit + 20*bonus 






class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

















class AgentMinimax(Agent):
    # TODO: section b : 1
    
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        curr_best = None
        
        def run_step_timeout():
            nonlocal curr_best
            tree_height = 0
            child_list = []
            child_ops, children_env = self.successors(env,agent_id)
            while True:
                child_list = [self.mini_max_RB(child_env, agent_id, tree_height, (agent_id+1)%2) for child_env in children_env]
                current_max = max(child_list)
                current_max_index = child_list.index(current_max)
                curr_best = child_ops[current_max_index]
                tree_height = tree_height + 1

        try:
            func_timeout(time_limit * 0.8, run_step_timeout, args=()) 
        except FunctionTimedOut:
            pass
        return curr_best
    
    def mini_max_RB(self, env: WarehouseEnv, agent_id, tree_height, turn):
        if env.done() or tree_height == 0:
            return smart_heuristic(env,agent_id)
        _ ,children_env = self.successors(env,turn)
        if turn == agent_id:
            current_max = (-1)*np.inf
            for child_env in children_env:
                val = self.mini_max_RB(child_env, agent_id, tree_height - 1, (turn+1)%2)
                current_max = max(current_max,val)
            return current_max
        else:
            current_min = np.inf
            for child_env in children_env:
                val = self.mini_max_RB(child_env, agent_id, tree_height - 1, (turn+1)%2)
                current_min = min(current_min,val)
            return current_min











class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        curr_best = None
        
        def run_step_timeout():
            nonlocal curr_best
            tree_height = 0
            child_list = []
            child_ops, children_env = self.successors(env,agent_id)
            while True:
                a = -np.inf
                b = np.inf
                child_list = [self.RB_alpha_beta(child_env, agent_id, tree_height, (agent_id+1)%2, a, b) for child_env in children_env]
                current_max = max(child_list)
                current_max_index = child_list.index(current_max)
                curr_best = child_ops[current_max_index]
                tree_height = tree_height + 1

        try:
            func_timeout(time_limit * 0.8, run_step_timeout, args=()) 
        except FunctionTimedOut:
            pass
        return curr_best
    
    def RB_alpha_beta(self, env: WarehouseEnv, agent_id, tree_height, turn, a, b):
        if env.done() or tree_height == 0:
            return smart_heuristic(env,agent_id)
        _ ,children_env = self.successors(env,turn)
        if turn == agent_id:
            current_max = (-1)*np.inf
            for child_env in children_env:
                val = self.RB_alpha_beta(child_env, agent_id, tree_height - 1, (turn+1)%2, a, b)
                current_max = max(current_max,val)
                a = max(current_max, a)
                if current_max >= b:
                    return np.inf
            return current_max
        else:
            current_min = np.inf
            for child_env in children_env:
                val = self.RB_alpha_beta(child_env, agent_id, tree_height - 1, (turn+1)%2, a, b)
                current_min = min(current_min,val)
                b = min(current_min, b)
                if current_min <= a:
                    return (-1)*np.inf
            return current_min











class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        curr_best = None
        
        def run_step_timeout():
            nonlocal curr_best
            tree_height = 0
            child_list = []
            child_ops, children_env = self.successors(env,agent_id)
            while True:
                child_list = [self.RB_expectimax(child_env, agent_id, tree_height, (agent_id+1)%2) for child_env in children_env]
                current_max = max(child_list)
                current_max_index = child_list.index(current_max)
                curr_best = child_ops[current_max_index]
                tree_height = tree_height + 1

        try:
            func_timeout(time_limit * 0.8, run_step_timeout, args=()) 
        except FunctionTimedOut:
            pass
        return curr_best

    def RB_expectimax(self, env: WarehouseEnv, agent_id, tree_height, turn):
        if env.done() or tree_height == 0:
            return smart_heuristic(env,agent_id)
        child_ops ,children_env = self.successors(env,turn)
        if turn == agent_id:
            current_max = (-1)*np.inf
            for child_env in children_env:
                val = self.RB_expectimax(child_env, agent_id, tree_height - 1, (turn+1)%2)
                current_max = max(current_max,val)
            return current_max
        else:
            num_ops = len(child_ops)
            sum_val = 0
            val = 0
            for op, child_env in zip(child_ops, children_env):
                if ("move east" in child_ops) and ("pick up" in child_ops):
                    if (op is "move east") or (op is "pick up"):
                        val = (2/(num_ops+2)) * self.RB_expectimax(child_env, agent_id, tree_height - 1, (turn+1)%2)
                    else:
                        val = (1/(num_ops+2)) * self.RB_expectimax(child_env, agent_id, tree_height - 1, (turn+1)%2)
                if ("move east" in child_ops) or ("pick up" in child_ops):
                    if (op is "move east") or (op is "pick up"):
                        val = (2/(num_ops+1)) * self.RB_expectimax(child_env, agent_id, tree_height - 1, (turn+1)%2)
                    else:
                        val = (1/(num_ops+1)) * self.RB_expectimax(child_env, agent_id, tree_height - 1, (turn+1)%2)
                else:
                    val = (1/(num_ops)) * self.RB_expectimax(child_env, agent_id, tree_height - 1, (turn+1)%2)
                sum_val = sum_val + val
            return sum_val










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