from torch.utils.data import Dataset
import torch
import os
import pickle
import copy
import numpy as np
from ridesharing.problems.ridesharing.state_ridesharing import StateRideSharing
from ridesharing.utils.beam_search import beam_search
from ridesharing.utils.functions import gps_distance

class RideSharing(object):
    NAME = 'ridesharing'

    @staticmethod
    def get_costs(dataset, obj, routes, schedule, engine):
        batch_size, graph_size = dataset['demand'].size()
        num_veh = dataset['capacity'].size(-1)

        total_time = []
        schedule_acc = copy.deepcopy(schedule)
        for i in range(batch_size):
            time = 0
            for j in range(num_veh):
                for k in range(1, len(schedule_acc[i][j])):
                    schedule_acc[i][j][k] += schedule_acc[i][j][k-1]
                    if k == len(schedule_acc[i][j]) - 1:
                        time += schedule_acc[i][j][k]
            total_time.append(time / 1000)

        total_delay_time = []
        total_remain_time = []
        veh_delay_time = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            remain_time = 0
            delay_time = 0
            for j in range(num_veh):
                if len(routes[i][j]) > 1:
                    for index, node in enumerate(routes[i][j]):
                        if node > num_veh:
                            # remain_time += dataset['time'][i, node - num_veh].item()
                            delay_time += max(0, schedule_acc[i][j][index - 1] - dataset['time'][i, node - num_veh].item())
                # veh_delay_time[i].append(delay_time / 1000)
            # total_delay_time.append(max(veh_delay_time[i]))
            total_delay_time.append(delay_time / graph_size)
        # if obj == 'min-max':
        #     return torch.max(total_dis, dim=1)[0], None
        # if obj == 'min-sum':
        #     return torch.sum(total_dis, dim=1), None
        # print("所有车辆行驶时间和：", total_time, "所有请求延误时间和：", total_delay_time)

        return torch.tensor(total_delay_time, dtype=dataset['time'].dtype, device=dataset['time'].device) #拼车实验
        # return torch.tensor(RideSharing.get_serve_number(graph_size, num_veh, routes), dtype=dataset['time'].dtype, device=dataset['time'].device)#派单实验

    @staticmethod
    def get_serve_number(graph_size,veh_size,veh_routes):
        num=0
        request_number=graph_size//2
        if(len(veh_routes)>=1):
            for i in range(0,len(veh_routes[0])):
                for j in range(0,len(veh_routes[0][i])):
                    if (veh_routes[0][i][j]>veh_size-1):
                        num+=1
        ready_to_serve=num%2
        num=num//2+ready_to_serve
        return graph_size//2-num

    @staticmethod
    def make_dataset(*args, **kwargs):
        return RideSharingDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateRideSharing.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = RideSharing.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    loc, demand, capacity, time, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float),  # scale demand
        'capacity': torch.tensor(capacity, dtype=torch.float),
        'time': torch.tensor(time, dtype=torch.float)
    }


class RideSharingDataset(Dataset):


    def __init__(self, filename=None, size=50, num_samples=10000, offset=0, distribution=None):
        super(RideSharingDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:
            WAIT_PICK_UP = 10  # wait to be pick up
            DETOUR_TIME = 5

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {          #不同顶点个数对应的车辆容量其实相同
                10: [20, 25, 30],
                20: [20, 25, 30],
                40: [20, 25, 30],
                50: [20, 25, 30],
                60: [20, 25, 30],
                80: [20, 25, 30],
                100: [20, 25, 30],
                120: [20, 25, 30],
            }

            num_veh = np.random.randint(3, 10)
            # num_veh = 4

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'veh_loc': torch.FloatTensor(num_veh, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': self.getdemand(size),
                    # 'capacity': torch.Tensor(CAPACITIES[size]),
                    'capacity': torch.Tensor(num_veh).uniform_(1, 5).int(),
                    'time': self.gettime(size, DETOUR_TIME),
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)  # self.size 表示数据个数  num_samples

    @staticmethod
    def getdemand(size):

        a = torch.FloatTensor(size).uniform_(0, 4).int() + 1        #一辆车最多四个人在这里暂时先设为四个人 但是车辆容量保持默认也就是10以上
        for i in range(int(len(a) / 2)):
            a[i + int(size / 2)] = a[i] * -1

        return a

    @staticmethod
    def gettime(size, DETOUR_TIME=None):
        time = torch.cat((torch.FloatTensor(size//2).uniform_(5, 15), torch.FloatTensor(size//2).uniform_(10, 30) + DETOUR_TIME), 0)
        return time

    def __len__(self):
        return self.size    # 返回数据个数

    def __getitem__(self, idx):
        return self.data[idx]  # index of sampled data


