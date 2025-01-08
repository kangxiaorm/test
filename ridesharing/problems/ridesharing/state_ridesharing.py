import logging

import torch
from typing import NamedTuple
from ridesharing.utils.boolmask import mask_long2bool, mask_long_scatter
from ridesharing.utils.functions import gps_distance, cosine
from ridesharing.engine import SimulatorEngine
import pandas as pd
import numpy as np
import copy

class StateRideSharing(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # veh + loc, [batch_size, veh + graph_size, 2]
    demand: torch.Tensor
    capacity: torch.Tensor
    to_visit: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    veh: torch.Tensor  # number of vehicles

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    to_delivery: torch.Tensor
    remain_time: torch.Tensor
    veh_passed_time: torch.Tensor

    cur_route: list
    cur_route_loc: list
    cur_route_time_cost: list
    cur_route_time_acc: list
    cur_route_time_allowance: list

    # VEHICLE_CAPACITY = [20, 25, 30]
    SPEED = 60 * 1000 / 3600    # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            # return self.visited_[:, None, :].expand(self.visited_.size(0), 1, -1).type(torch.ByteTensor)
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):  # coords: []
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                veh=self.veh[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                to_delivery=self.to_delivery[key],
                remain_time=self.remain_time[key],
                veh_passed_time=self.veh_passed_time[key]
            )
        # return super(StateRideSharing, self).__getitem__(key)

    @staticmethod
    def initialize(input, engine, visited_dtype=torch.uint8):

        loc = input['loc']
        veh = input['veh']
        demand = input['demand']
        capacity = input['capacity']
        time = input['time']
        to_visit = input['to_visit']

        batch_size, n_loc, _ = loc.size()  # n_loc = graph_size
        num_veh = capacity.size(-1)

        to_delivery = torch.cat([torch.ones(batch_size, num_veh, n_loc // 2, dtype=torch.uint8, device=loc.device),
                                 torch.zeros(batch_size, num_veh, n_loc // 2, dtype=torch.uint8, device=loc.device)],
                                dim=-1)

        visited_ = torch.zeros(batch_size, 1, n_loc, dtype=torch.uint8, device=loc.device)

        # Set the previous requests as visited
        graph = engine.strategy.waiting_requests_batch
        for batch_id, reqs in enumerate(graph):
            for req in reqs:
                if req['status'] == 'assigned':
                    node_index = reqs.index(req)
                    visited_[batch_id, 0, node_index] = 1
                    visited_[batch_id, 0, node_index + n_loc // 2] = 1

        # init the route, schedule and time of all vehicles
        cur_route = [[[] for _ in range(num_veh)] for _ in range(batch_size)]
        cur_route_loc = [[[] for _ in range(num_veh)] for _ in range(batch_size)]
        cur_route_time_cost = [[[] for _ in range(num_veh)] for _ in range(batch_size)]
        # 路线累计行驶时间
        cur_route_time_acc = [[[] for _ in range(num_veh)] for _ in range(batch_size)]
        # 路线时间余量
        cur_route_time_allowance = [[[] for _ in range(num_veh)] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(num_veh):
                cur_route[i][j].append(j)
                cur_route_loc[i][j].append(veh[i, j, :])
                # 判断车辆是否存在计划路线并添加路线和时间
                vh = engine.vehicle_manager.vehicles[engine.strategy.candidate_vehicles_batch[j]]
                if vh.schedule:
                    for sc in vh.schedule:
                        try:
                            psg_index = engine.strategy.waiting_requests_id_batch.index(sc[0])
                        except Exception as e:
                            engine.strategy.logger.exception(e)
                            continue
                        if sc[2] == 0:  # pickup
                            cur_route[i][j].append(psg_index + num_veh)
                            cur_route_loc[i][j].append(loc[i, psg_index, :])
                        else:   # delivery
                            cur_route[i][j].append(psg_index + n_loc // 2 + num_veh)
                            cur_route_loc[i][j].append(loc[i, psg_index + n_loc // 2, :])
                if len(cur_route_loc[i][j]) > 1:
                    for k in range(0, len(cur_route_loc[i][j])-1):
                        source_node = 0
                        target_node = 0
                        if k < 1:
                            source_node, target_node = get_vh_source_and_target(engine, j, cur_route[i][j][k+1] - num_veh)
                        else:
                            source_node, target_node = get_source_and_target(engine, cur_route[i][j][k] - num_veh, cur_route[i][j][k+1] - num_veh)
                        _, time_cost = engine.vehicle_manager.shortest_travel_path_cost(
                            source_node,
                            target_node,
                            engine.timestamp
                        )
                        if time_cost is None:
                            time_cost = 10000
                            engine.strategy.logger.info(
                                f"source_node:{source_node} -> target_node:{target_node} cost is None")
                        cur_route_time_cost[i][j].append(time_cost)

        for i in range(batch_size):
            for j in range(num_veh):
                if len(cur_route_time_cost[i][j]) > 0:
                    cur_route_time_acc[i][j].append(cur_route_time_cost[i][j][0])
                    cur_route_time_allowance[i][j].append(time[i, cur_route[i][j][1] - num_veh].item() - cur_route_time_acc[i][j][0])
                for k in range(1, len(cur_route_time_cost[i][j])):
                    cur_route_time_acc[i][j].append(cur_route_time_cost[i][j][k] + cur_route_time_cost[i][j][k-1])
                    cur_route_time_allowance[i][j].append(time[i, cur_route[i][j][k+1] - num_veh].item() - cur_route_time_acc[i][j][k])

        return StateRideSharing(
            coords=torch.cat((veh, loc), -2),
            demand=demand,
            capacity=capacity,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            veh=torch.arange(num_veh, dtype=torch.int64, device=loc.device)[:, None],
            # prev_a is current node
            # prev_a=torch.zeros(batch_size, num_veh, dtype=torch.long, device=loc.device), # 不应该为0，0代表depot 这里没有depot 应该是每辆车的当前所在节点先设置为-1
            prev_a=torch.arange(0, num_veh, dtype=torch.long, device=loc.device).repeat(batch_size, 1), # ?
            used_capacity=demand.new_zeros(batch_size, num_veh),
            veh_passed_time=torch.zeros(batch_size, num_veh, dtype=torch.long, device=loc.device),  # ?
            remain_time=time,   # ?
            visited_ = visited_,
            lengths=torch.zeros(batch_size, num_veh, device=loc.device),    # ?
            cur_coord=input['veh'], # ?
            # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            to_delivery=to_delivery,
            to_visit=to_visit,   # Combination of requests and vehicles
            cur_route=cur_route,
            cur_route_loc=cur_route_loc,
            cur_route_time_cost=cur_route_time_cost,
            cur_route_time_acc=cur_route_time_acc,
            cur_route_time_allowance=cur_route_time_allowance,
        )

    def get_final_cost(self):
        assert self.all_finished()
        # coords: [batch_size, graph_size+1, 2]
        return self.lengths  # + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected, veh, engine):  # [batch_size, num_veh]
        '''

        Args:
            selected: type: list 代表每个batch的选择，格式[batch_id, insert_node, insert_index]
            veh: size:[batch_size] 每个batch所选的车辆
            engine:

        Returns:

        '''
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        batch_size, num_veh = self.capacity.size()
        n_loc = self.to_delivery.size(-1)  # number of customers

        new_to_delivery = torch.zeros_like(veh)
        for i in range(0, veh.size(-1)):
            if selected[i][1] < n_loc // 2:
                new_to_delivery[i] = selected[i][1] + n_loc // 2  # the pair node of selected node
            else:
                new_to_delivery[i] = selected[i][1] - n_loc // 2

        # remain_time = self.remain_time
        # for i in range(0, veh.size(-1)):
        #     remain_time[i, selected[i][1]] = 0

        used_capacity = self.used_capacity
        for i in range(0, veh.size(-1)):
            # add capacity
            if self.demand[i, selected[i][1]] > 0: # 如果是d点，容量不能加1; 因为所有乘客是一次性派给车辆的，车辆的最大容量为4
                used_capacity[i, veh[i]] += self.demand[i, selected[i][1]]
            # set visited node 1
            self.visited_[i, :, selected[i][1]] = 1
            # modify pickup and delivery relation
            self.to_delivery[i, veh[i], selected[i][1]] = 0
            self.to_delivery[i, veh[i], new_to_delivery[i]] = 1

        visited_ = self.visited_
        to_delivery = self.to_delivery

        for i in range(0, veh.size(-1)):
            insert_node = selected[i][1]
            insert_index = selected[i][2]
            route_length = len(self.cur_route[i][veh[i]])

            if insert_index < 1:
                source_node, target_node = get_vh_source_and_target(engine, self.cur_route[i][veh[i]][insert_index], insert_node)
            else:
                source_node, target_node = get_source_and_target(
                    engine, 
                    self.cur_route[i][veh[i]][insert_index] - num_veh,
                    insert_node)
                
            _, time_cost_pre = engine.vehicle_manager.shortest_travel_path_cost(source_node, target_node, engine.timestamp)

            if time_cost_pre is None:
                time_cost_pre = 10000
                engine.strategy.logger.info(f"source_node:{source_node} -> target_node:{target_node} cost is None")

            self.cur_route[i][veh[i]].insert(insert_index + 1, insert_node.item() + num_veh) # 将选择的节点插入路线中
            self.cur_route_loc[i][veh[i]].insert(insert_index + 1, self.coords[i, insert_node + num_veh, :]) # 将选择的节点坐标插入路线中
            if insert_index >= route_length - 1: # 插入位置为末尾位置
                self.cur_route_time_cost[i][veh[i]].append(time_cost_pre)
                if len(self.cur_route_time_acc[i][veh[i]]) > 0:
                    self.cur_route_time_acc[i][veh[i]].append(time_cost_pre + self.cur_route_time_acc[i][veh[i]][insert_index - 1])
                else:
                    self.cur_route_time_acc[i][veh[i]].append(time_cost_pre)
                # print("remain_time:", self.remain_time[i, insert_node].item(), "travel_time", self.cur_route_time_acc[i][veh[i]][insert_index])
                self.cur_route_time_allowance[i][veh[i]].append(
                    self.remain_time[i, insert_node].item() - self.cur_route_time_acc[i][veh[i]][insert_index])
            else:
                self.cur_route_time_cost[i][veh[i]][insert_index] = time_cost_pre
                self.cur_route_time_acc[i][veh[i]][insert_index] = \
                    time_cost_pre + self.cur_route_time_acc[i][veh[i]][insert_index - 1]
                self.cur_route_time_allowance[i][veh[i]][insert_index] = \
                    self.remain_time[i, insert_node].item() - self.cur_route_time_acc[i][veh[i]][insert_index]
                new_source_node, new_target_node = get_source_and_target(engine, insert_node, self.cur_route[i][veh[i]][insert_index + 2] - num_veh)
                _, time_cost_last = engine.vehicle_manager.shortest_travel_path_cost(
                    new_source_node,
                    new_target_node,
                    engine.timestamp
                )
                if time_cost_last is None:
                    time_cost_last = 10000
                    engine.strategy.logger.info(f"source_node:{new_source_node} -> target_node:{new_target_node} cost is None")

                self.cur_route_time_cost[i][veh[i]].insert(insert_index + 1, time_cost_last)
                self.cur_route_time_acc[i][veh[i]].insert(insert_index + 1,
                                                          self.cur_route_time_acc[i][veh[i]][insert_index] + time_cost_last)
                self.cur_route_time_allowance[i][veh[i]].insert(insert_index + 1,
                     self.remain_time[i, self.cur_route[i][veh[i]][insert_index + 2] - num_veh].item() - self.cur_route_time_acc[i][veh[i]][insert_index + 1])

            for k in range(insert_index+2, len(self.cur_route_time_cost[i][veh[i]])):
                self.cur_route_time_acc[i][veh[i]][k] = self.cur_route_time_cost[i][veh[i]][k] + self.cur_route_time_acc[i][veh[i]][k-1]
                self.cur_route_time_allowance[i][veh[i]][k] = self.remain_time[i, self.cur_route[i][veh[i]][k+1] - num_veh].item() - \
                                                              self.cur_route_time_acc[i][veh[i]][k]
        return self._replace(
            used_capacity=used_capacity, visited_=visited_, i=self.i + 1, to_delivery=to_delivery
        )

    def delete_invalid_source_node(self, batch_id, veh, target_node, engine):
        graph_size = self.demand.size(-1)
        num_veh = self.capacity.size(-1)

        self.visited_[batch_id, 0, target_node] = 1
        source_node = target_node - graph_size//2 + num_veh

        route_length = len(self.cur_route[batch_id][veh])
        source_node_index = self.cur_route[batch_id][veh].index(source_node)

        self.cur_route[batch_id][veh].pop(source_node_index)
        self.cur_route_loc[batch_id][veh].pop(source_node_index)

        if source_node_index >= route_length - 1:
            self.cur_route_time_cost[batch_id][veh].pop()
            self.cur_route_time_acc[batch_id][veh].pop()
            self.cur_route_time_allowance[batch_id][veh].pop()
        else:
            pre_node = self.cur_route[batch_id][veh][source_node_index - 1]
            next_node = self.cur_route[batch_id][veh][source_node_index]
            if pre_node < num_veh:
                source_node_id, target_node_id = get_vh_source_and_target(engine, pre_node, next_node - num_veh)
            else:
                source_node_id, target_node_id = get_source_and_target(engine, pre_node - num_veh, next_node - num_veh)
            _, time_cost = engine.vehicle_manager.shortest_travel_path_cost(source_node_id, target_node_id, engine.timestamp)

            if time_cost is None:
                time_cost = 10000.0
                engine.strategy.logger.info(f"source_node:{source_node_id} -> target_node:{target_node_id} cost is None")

            self.cur_route_time_cost[batch_id][veh].pop(source_node_index - 1)
            self.cur_route_time_cost[batch_id][veh][source_node_index - 1] = time_cost

            self.cur_route_time_acc[batch_id][veh].clear()
            self.cur_route_time_allowance[batch_id][veh].clear()

            for i in range(len(self.cur_route_time_cost[batch_id][veh])):
                if i == 0:
                    self.cur_route_time_acc[batch_id][veh].append(self.cur_route_time_cost[batch_id][veh][i])
                else:
                    self.cur_route_time_acc[batch_id][veh].append(self.cur_route_time_acc[batch_id][veh][i - 1] +
                                                                  self.cur_route_time_cost[batch_id][veh][i])
                self.cur_route_time_allowance[batch_id][veh].append(
                    self.remain_time[batch_id, self.cur_route[batch_id][veh][i+1] - num_veh].item() -
                    self.cur_route_time_acc[batch_id][veh][i])

    def get_avail_pos(self, veh, engine):
        """
        Gets a (batch_size, n_loc ) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        veh： 当前时刻所有batch选择的车
        :return:
        """
        batch_size, graph_size = self.demand.size()
        num_veh = self.capacity.size(-1)
        cur_route = self.get_cur_route()
        cur_route_loc = self.cur_route_loc
        cur_route_time_cost = self.cur_route_time_cost
        cur_route_time_acc = self.cur_route_time_acc
        cur_route_time_allowance = self.cur_route_time_allowance

        # alpha = 0.707
        alpha = 0.999999921837146

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, :]  # [batch_size, 1, n_loc]
        else:
            visited_loc = self.visited_[:, :][:, None, :]  # [batch_size, 1, n_loc]

        to_visited_loc = self.to_visit.clone()

        invalid_demand = []
        exceeds_cap = []
        delivery_for_veh = []
        mask = []
        final_avail_pos = []
        for i in range(0, batch_size):
            # capacity
            plan_route = cur_route[i][veh[i]]
            plan_route_loc = cur_route_loc[i][veh[i]]
            plan_route_time = cur_route_time_cost[i][veh[i]]
            plan_route_acc_time = cur_route_time_acc[i][veh[i]]
            plan_route_time_allowance = cur_route_time_allowance[i][veh[i]]

            plan_route_length = len(plan_route)

            cur_capacity = (self.capacity[i, veh[i]] - self.used_capacity[i, veh[i]]).repeat(plan_route_length, 1).squeeze(1)
            # for k in range(1, plan_route_length):
            #     if (plan_route[k] >= (num_veh + graph_size // 2)) and (plan_route[k] < (num_veh + graph_size)):
            #         cur_capacity[k] = cur_capacity[k - 1] + 1

            # demand == 0
            invalid_demand.append((self.demand[self.ids[i], :] == 0)[..., None].repeat(1, 1,
                                                                             plan_route_length))  # [1, graph_size, plan_route_length]
            # demand > remain capacity
            route_demand = self.demand[self.ids[i], :].unsqueeze(-1).expand(-1, -1, plan_route_length)
            exceeds_cap.append(route_demand > cur_capacity[None, None, :].expand_as(route_demand))

            # is_to_delivery: delivery node can only be inserted after the corresponding pickup node
            cur_delivery = self.to_delivery[i, veh[i], :][None, :, None].repeat(1, 1, plan_route_length)
            pick_index = (1 - self.to_delivery[i, veh[i], :])[0:graph_size // 2].nonzero().squeeze(-1).tolist()
            for pick_node in pick_index:
                try:
                    k = plan_route.index(pick_node + num_veh)
                except Exception as e:
                    engine.strategy.logger.exception(e)
                    engine.strategy.logger.info(f"element:{pick_node + num_veh} not in list:{plan_route}")
                    continue
                cur_delivery[:, pick_node + graph_size // 2, 0:k] = 0
            delivery_for_veh.append(1 - cur_delivery)

            # is_visited
            is_visited = visited_loc[i].unsqueeze(-1).expand(-1, -1, plan_route_length)

            # is_to_visit
            to_visit = to_visited_loc[i, veh[i], :][None, :, None].expand(-1, -1, plan_route_length)
            to_visit[i, graph_size // 2:, :] = 0

            # delay_time
            delay_time = torch.ones(plan_route_length, dtype=to_visit.dtype, device=to_visit.device)
            delay_time[plan_route_length-1] = 0 # 最后一个位置默认可用
            for k in range(len(plan_route_time_allowance)-1, -1, -1):
                if plan_route_time_allowance[k] < 0:
                    break
                else:
                    delay_time[k] = 0
            delay_time = delay_time[None, None, :].expand(-1, graph_size, -1)

            mask_pos = is_visited | invalid_demand[i] | exceeds_cap[i] | delivery_for_veh[i] | to_visit | delay_time

            # time
            avail_pos = (1 - mask_pos).nonzero()
            invalid_target = {}
            for k in range(0, avail_pos.size(0)):  # [batch_id, node_index, insert_position]
                insert_pos = avail_pos[k, 2]
                insert_node = avail_pos[k, 1]  # coords中的坐标包含车辆的初始结点

                invalid_target[insert_node.item()] = 0

                if plan_route[insert_pos] < num_veh:
                    source_node, target_node = get_vh_source_and_target(engine, plan_route[insert_pos], insert_node)
                else:
                    source_node, target_node = get_source_and_target(engine, plan_route[insert_pos] - num_veh, insert_node)
                _, time_cost = engine.vehicle_manager.shortest_travel_path_cost(source_node, target_node, engine.timestamp)

                if time_cost is None:
                    mask_pos[:, insert_node, insert_pos] = 1
                    if insert_node >= graph_size // 2:
                        invalid_target[insert_node.item()] += 1
                        if invalid_target[insert_node.item()] == (1 - mask_pos[i, insert_node, :]).nonzero().size(0):  # 无效位置等于可用位置
                            # self.delete_invalid_source_node(i, veh[i], insert_node, engine) # d点不可用，删除路线中对应的o点
                            pass
                else:
                    if insert_pos > 0:
                        time_cost += plan_route_acc_time[insert_pos - 1]
                    if insert_node < graph_size // 2 and time_cost > self.remain_time[i, insert_node]:
                        mask_pos[:, insert_node, insert_pos] = 1

            # similarity
            if plan_route_length > 1:
                avail_pos = (1 - mask_pos).nonzero()
                mobility_vector = []
                # mobility_vector.append(plan_route_loc[0])
                for k, loc in enumerate(plan_route_loc):
                    # 添加路线中所有o点对应的d点坐标
                    if k > 0:
                        if plan_route[k] >= num_veh + graph_size // 2:   # d点
                            mobility_vector.append(self.coords[i, plan_route[k], :])
                        elif (plan_route[k] + graph_size // 2) not in plan_route:   # 是o点 且d点还没选择
                            mobility_vector.append(self.coords[i, plan_route[k] + graph_size // 2, :])
                # 对mobility_vector取平均
                mobility_vector = [torch.stack(mobility_vector).mean(0)]
                mobility_vector.insert(0, plan_route_loc[0])
                unavail_node = []
                for k in range(0, avail_pos.size(0)):
                    requests_vector = []
                    insert_pos = avail_pos[k, 2]
                    insert_node = avail_pos[k, 1]
                    if insert_node.item() in unavail_node:
                        continue
                    if insert_node < graph_size // 2:
                        requests_vector.append(self.coords[i, insert_node, :])
                        requests_vector.append(self.coords[i, insert_node + graph_size // 2, :])
                        cos = cosine(mobility_vector, requests_vector)
                        if cos < alpha: # 这里的关系
                            mask_pos[:, insert_node, insert_pos] = 1
                            unavail_node.append(insert_node.item())

            mask.append(mask_pos)

        for mask_pos in mask:
            final_avail_pos.append((1 - mask_pos).nonzero())

        # remove unavailable actions
        final_avail_pos = self.filter_pos_by_schedule(final_avail_pos, veh, engine)

        return final_avail_pos

    def filter_pos_by_schedule(self, final_avail_pos, veh, engine):
        batch_size, graph_size = self.demand.size()
        # batch_size = len(final_avail_pos)
        num_veh = self.capacity.size(-1)

        for i in range(0, batch_size):
            sc = self.cur_route_time_cost[i][veh[i]].copy() # 节点间消耗的时间
            route = self.cur_route[i][veh[i]].copy() # 当前车辆的路线
            tw = {} # 当前车辆路线上每个节点的剩余时间
            mask = []
            if len(route) > 1:
                for node in route[1:]:
                    tw[node] = self.remain_time[i, node-num_veh].item() #路线中的节点都是加了车辆数目之后的
            for ii in range(0, final_avail_pos[i].size(0)):
                update_sc = sc.copy()
                update_route = route.copy()
                update_tw = tw.copy()
                insert_node, insert_pos = final_avail_pos[i][ii][1].item(), final_avail_pos[i][ii][2].item()
                update_route.insert(insert_pos+1, insert_node+num_veh)
                update_tw[insert_node+num_veh] = self.remain_time[i, insert_node].item()

                cur_node = update_route[0]
                ts = engine.timestamp
                a_cost = 0
                flag = False
                if insert_node < graph_size // 2:
                    for iii in range(1, len(update_route)):
                        next_node = update_route[iii]
                        source_node, target_node = cur_node, next_node
                        if cur_node < num_veh:
                            source_node, target_node = get_vh_source_and_target(engine, cur_node, next_node - num_veh)
                        else:
                            source_node, target_node = get_source_and_target(engine, cur_node - num_veh, next_node - num_veh)
                        _, cost = engine.vehicle_manager.shortest_travel_path_cost(source_node, target_node, ts)
                        if cost is None:
                            flag = True
                            break
                        a_cost += cost

                        if a_cost > update_tw[next_node]:
                            flag = True
                            break

                        cur_node = next_node
                        ts += pd.Timedelta(cost, unit='s')
                mask.append(flag)
            if any(mask):
                indices = torch.tensor([j for j in range(len(mask)) if mask[j] == False], dtype=torch.int64, device=self.capacity.device)
                final_avail_pos[i] = torch.index_select(final_avail_pos[i], 0, indices)

        return final_avail_pos

    def get_mask_veh(self, invalid_veh):
        batch_size, num_veh = self.capacity.size()
        graph_size = self.demand.size(-1)

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, :]  # [batch_size, 1, n_loc]
        else:
            visited_loc = self.visited_[:, :][:, None, :]  # [batch_size, 1, n_loc]

        mask = torch.zeros_like(self.capacity)  # [batch_size, veh_num]
        for i in range(0, batch_size):
            for j in range(0, num_veh):
                if (self.to_delivery[i, j, visited_loc.size(-1) // 2:] == 0).all():
                    if (self.capacity[i, j] - self.used_capacity[i, j] < self.demand[i, 0:graph_size//2]).all():  # 该车没有满足容量的o点
                        mask[i, j] = 1
                    elif self.to_visit[i, j, :].all():  # 该车没有可以到达的o点
                        mask[i, j] = 1
                    elif visited_loc[i, :, (1 - self.to_visit[i, j, :]).nonzero().squeeze(0)].all(): # 该车可以到达的所有o点已被访问
                        mask[i, j] = 1
                    else:
                        pass
                else:
                    pass

        for veh in invalid_veh:
            mask[0, veh] = 1

        return mask > 0

    def construct_solutions(self, actions):
        return actions

    def all_finished(self):
        return self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_cur_node(self):
        return self.prev_a

    def get_cur_route(self):
        return self.cur_route

def get_source_and_target(engine, source_index, target_index):
    source_node = 0
    target_node = 0
    graph_size = len(engine.strategy.waiting_requests_batch[0]) * 2
    if source_index < graph_size // 2:
        source_node = engine.strategy.waiting_requests_batch[0][source_index]['source']
    else:
        source_node = engine.strategy.waiting_requests_batch[0][source_index - graph_size // 2]['target']
    if target_index < graph_size // 2:
        target_node = engine.strategy.waiting_requests_batch[0][target_index]['source']
    else:
        target_node = engine.strategy.waiting_requests_batch[0][target_index - graph_size // 2]['target']
    
    return source_node, target_node

def get_vh_source_and_target(engine, veh_index, target_index):
    source_node = 0
    target_node = 0
    vh = engine.vehicle_manager.vehicles[engine.strategy.candidate_vehicles_batch[veh_index]]
    source_node = vh.cur_node

    graph_size = len(engine.strategy.waiting_requests_batch[0]) * 2
    if target_index < graph_size // 2:
        target_node = engine.strategy.waiting_requests_batch[0][target_index]['source']
    else:
        target_node = engine.strategy.waiting_requests_batch[0][target_index - graph_size // 2]['target']

    return source_node, target_node
