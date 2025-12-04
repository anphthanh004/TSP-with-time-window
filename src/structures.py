import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass

# Problem chỉ chứa dữ liệu đề bài
@dataclass(frozen=True) 
class Problem:
    time_matrix: np.ndarray 
    request: list           
    num_request: int        
    type: str = 'GA'        
    penalty: float = 1000.0 

class Individual:
    def __init__(self, problem=None):
        """
        Individual chỉ tham chiếu (reference) tới Problem => quan hệ Association (liên kết)
        Individual ---(Association)---> Problem
        """   
        self.route = []
        self.problem = problem
        self.fitness = None
        self.route_computing = None
        
    def copy(self):
        new_ind = Individual(self.problem)
        new_ind.route = self.route[:]
        return new_ind
        
    def compute_route_forward(self, route, problem):
        n = len(route)
        arrivals = [0]*n
        departures = [0]*n
        lateness = [0]*n
        wait = [0]*n
        
        current_time = 0
        prev = 0
        for idx in range(n):
            node = route[idx]
            e_i, l_i, d_i = problem.request[node-1]
            travel = problem.time_matrix[prev][node]
            arrival = current_time + travel
            # waiting allowed: arrive earlier -> wait
            if arrival < e_i:
                arrival = e_i
            # lateness allowed: measure how much late
            late = max(0.0, arrival - l_i)
            wait.append(max(0.0, e_i - arrival))
            arrivals[idx] = arrival
            lateness[idx] = late
            departures[idx] = arrival + d_i
            current_time = departures[idx]
            prev = node

        # return to depot
        current_time += problem.time_matrix[prev][0]
        total_time = current_time
        total_lateness = sum(lateness)
        total_wait = sum(wait)
        return arrivals, departures, total_time, lateness, total_lateness, wait, total_wait

    def calObjective(self, problem):
        #---------------------
        # Rest mỗi khi gọi lại (có thể lúc này nó thuộc quần thể mới)
        self.objective = None
        self.route_computing = None
        
        
        if self.problem.type == 'MOO': 
            self.distance = None
            self.rank = None
        #----------------------
        
        # 0-arrivals, 1-departures, 2-total_time, 3-lateness, 4-total_lateness, 5-wait, 6-total_wait
        self.route_computing = self.compute_route_forward(self.route, problem)
        if self.problem.type == 'GA':
            total_time = self.route_computing[2]
            total_lateness = self.route_computing[4]
            self.objective = total_time + problem.penalty * total_lateness
            return self.objective
        if self.problem.type == 'MOO':
            total_time = self.route_computing[2]
            lateness = self.route_computing[3]
            total_lateness = self.route_computing[4]
            nums_of_late_arrivals = sum(1 for x in lateness if x != 0.0)
            # thời gian hoàn thành, số lần đến muộn, tổng thời gian đến muộn
            self.objective = (total_time, nums_of_late_arrivals, total_lateness)
        

