import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import pandas as pd
from typing import NamedTuple
from ridesharing.utils.tensor_functions import compute_in_batches

from ridesharing.nets.graph_encoder import GraphAttentionEncoder
from ridesharing.engine import SimulatorEngine
from torch.nn import DataParallel
from ridesharing.utils.beam_search import CachedLookup
from ridesharing.problems.ridesharing.state_ridesharing import get_source_and_target, get_vh_source_and_target
from ridesharing.utils.functions import save_result
import copy
import datetime
import random

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    # glimpse_key: torch.Tensor
    # glimpse_val: torch.Tensor
    # logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key]
                # glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                # glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                # logit_key=self.logit_key[key]
            )
        # return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 obj,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.obj = obj
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_ridesharing = problem.NAME == 'ridesharing'
        self.feed_forward_hidden = 512

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_ridesharing:
            # Embedding of last node + remaining_capacity +remain time/ remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1 + 1
            # num_veh = 3
            node_dim = 2 + 1 + 1  # x,y, demand(3 vehicles)+remain time
            veh_dim = 3  # 三个坐标 三个距离 三个pass_time没加
            self.FF_time = nn.Sequential(
                nn.Linear(1, self.embedding_dim),
                nn.Linear(self.embedding_dim, self.feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(self.feed_forward_hidden, self.embedding_dim)
            ) if self.feed_forward_hidden > 0 else nn.Linear(self.embedding_dim, self.embed_dim)

            self.FF_tour = nn.Sequential(
                # nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Linear(self.embedding_dim, self.feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(self.feed_forward_hidden, self.embedding_dim)
            ) if self.feed_forward_hidden > 0 else nn.Linear(self.embedding_dim, self.embed_dim)
            self.select_embed = nn.Linear(self.embedding_dim * 2, 1)
            self.init_embed_pick = nn.Linear(node_dim * 2, embedding_dim)
            self.init_embed_delivery = nn.Linear(node_dim, embedding_dim)

            # Special embedding projection for depot node
            self.init_embed_veh = nn.Linear(2, embedding_dim)
            # self.init_embed_ret = nn.Linear(2 * embedding_dim, embedding_dim)

        self.init_embed = nn.Linear(node_dim, embedding_dim)  # node_embedding

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_route_embeddings = nn.Linear(embedding_dim * 16, embedding_dim, bias=False)
        self.project_insert_route_embeddings = nn.Linear(embedding_dim * 16, embedding_dim, bias=False)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)


    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, engine: SimulatorEngine, return_pi=True):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results_1114_1120 may be of different lengths on different GPUs
        :return:
        """
        num_veh = input['capacity'].size(-1)
        graph_size = input['demand'].size(-1)
        if num_veh != 0 and graph_size != 0:
            # embeddings: [batch_size, graph_size+1, embed_dim]
            if self.checkpoint_encoder:
                embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
            else:
                embeddings, _ = self.embedder(self._init_embed(input))  # embeddings: [batch_size, veh_size + graph_size, embed_dim]

            # 每一步的可用位置 每一步可用位置的概率分布 每一步所选车辆 每一步对应车辆所选操作 每辆车最终路线 每辆车最终路线花费时间
            avail_pos, _log_p, sequence, sequence_log_p, veh_list, routes, schedule, _log_p_veh= \
                self._inner(input, embeddings, engine)
        else:
            routes = []
            schedule = [] # 路线花费时间
            sequence_log_p = []
            veh_list = []
            _log_p_veh = []

        cost = self.problem.get_costs(input, self.obj, routes, schedule, engine)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll, ll_veh = self._calc_log_likelihood(input, sequence_log_p, veh_list, _log_p_veh)  # [batch_size]   ll, ll_veh是什么意思？
        if return_pi:
            return cost, ll, ll_veh, routes

        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, input, _log_p, veh_list, _log_p_veh):
        if len(_log_p) == 0:
            _log_p.append([torch.tensor([0.], dtype=torch.float32, device=input['loc'].device, requires_grad=True)])
        for i in range(len(_log_p)):
            _log_p[i] = torch.stack(_log_p[i])
        log_p = torch.stack(_log_p).squeeze(-1).sum(0)

        if len(veh_list) != 0:
            log_p_veh = _log_p_veh.gather(2, torch.tensor(veh_list).cuda().unsqueeze(-1)).squeeze(-1)
            log_p_veh = log_p_veh.sum(1)
        else:
            log_p_veh = torch.tensor([0.], dtype=torch.float32, device=input['loc'].device, requires_grad=True)

        # if not (log_p > -1000).data.all():
        #     print("_calc_log_likelihood(): ", (log_p > -1000).data.all())
        # assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        # assert (log_p_veh > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        # return log_p.sum(1), log_p_veh.sum(1)  # [batch_size]
        return log_p, log_p_veh

    def _init_embed(self, input):

        if self.is_ridesharing:
            batch_size, n_loc, _ = input['loc'].size()
            # n_veh = input['veh_loc'].size(1)
            # n_loc = n - n_veh
            features = ('demand', 'time')

            # capa = input['capacity'][:, None].expand(input['capacity'].size(0), input['loc'].size(1)+1, input['capacity'].size(-1))

            # normalize demand
            # demand = torch.tensor([(input['demand'] / input['capacity'][0:1, veh]).tolist() for veh in
            #                        range(input['capacity'].size(-1))]).transpose(0, 1).transpose(1, 2).cuda()      # [batch_size, graph_size, num_veh]

            pick_with_feature_1 = torch.cat((  # [batch_size, graph_size//2, 4]
                        input['loc'][:, :n_loc // 2, :],  # [batch_size, graph_size//2, 2]
                        input['demand'][:, :n_loc // 2, None],  # [batch_size, graph_size//2, 1]
                        input['time'][:, :n_loc // 2, None],   # [batch_size, graph_size//2, 1]
                    ), -1)

            pick_with_feature_2 = torch.cat((  # [batch_size, graph_size//2, 4]
                        input['loc'][:, n_loc // 2:, :],  # [batch_size, graph_size//2, 2]
                        input['demand'][:, n_loc // 2:, None],  # [batch_size, graph_size//2, 1]
                        input['time'][:, n_loc // 2:, None],  # [batch_size, graph_size//2, 1]
                    ), -1)

            # vehicles node embedding
            # pick_with_feature_3 = torch.cat((
            #             input['loc'][:, n_loc:, :],
            #             torch.zeros(batch_size, n_veh, dtype=torch.float32, device=input['loc'].device)[..., None],
            #             torch.zeros(batch_size, n_veh, dtype=torch.float32, device=input['loc'].device)[..., None],
            #         ), -1)

            feature_pick = torch.cat([pick_with_feature_1, pick_with_feature_2],
                                     -1)  # [batch_size, graph_size//2, 8]

            feature_delivery = pick_with_feature_2
            # feature_delivery = torch.cat([pick_with_feature_2, pick_with_feature_3], 1)  # [batch_size, graph_size//2, 4]

            embed_veh = self.init_embed_veh(input['veh'])
            embed_pick = self.init_embed_pick(feature_pick)
            embed_delivery = self.init_embed_delivery(feature_delivery)

            return torch.cat([embed_veh, embed_pick, embed_delivery], 1) #[batchsize,graphsize,embedding]

    def select_veh(self, input, state, embeddings, mask_veh):
        '''
            input: mask_veh [batch_size, num_veh]   False/True
        '''
        batch_size, _, embed_dim = embeddings.size()
        num_veh = state.capacity.size(-1)

        route_embed = torch.zeros([batch_size, num_veh, embed_dim], device=embeddings.device, dtype=embeddings.dtype)
        time_allowance_embed = torch.zeros([batch_size, num_veh, 1], device=embeddings.device, dtype=embeddings.dtype)
        for i in range(batch_size):
            for j in range(num_veh):
                if len(state.cur_route_time_allowance[i][j]) != 0:
                    time_allowance_embed[i, j, :] = torch.tensor(state.cur_route_time_allowance[i][j],
                                                        device=embeddings.device, dtype=embeddings.dtype).mean()

                route = torch.tensor(state.cur_route[i][j], device=embeddings.device, dtype=torch.int64)

                route_embed[i, j, :] = torch.gather(
                    embeddings[i, :, :],
                    0,
                    route.contiguous().view(route.size(-1), 1).expand(-1, embeddings.size(-1))
                ).mean(0)

        route_context = self.FF_tour(route_embed)     # [batch_size, num_veh, 128]
        time_allowance_context = self.FF_time(time_allowance_embed)   # [batch_size, num_veh, 128]
        context = torch.cat((route_context, time_allowance_context), -1).view(batch_size, num_veh, self.embedding_dim * 2)   # 需要修改维度 view()函数 [batch_size, num_veh, 256]

        log_veh = F.log_softmax(self.select_embed(context), dim=1).squeeze(-1)  # [batch_size, num_veh]
        if self.decode_type == "greedy":
            logits = F.softmax(self.select_embed(context), dim=1).squeeze(-1)
            logits[mask_veh] = -math.inf
            veh = torch.max(logits.exp(), dim=1)[1]
        elif self.decode_type == "sampling":
            logits = F.softmax(self.select_embed(context), dim=1).squeeze(-1)   # [batch_size, num_veh]
            logits[mask_veh] = -math.inf
            try:
                veh = logits.exp().multinomial(1).squeeze(-1)   # multinomial()函数
            except:
                print("error")
        return veh, log_veh.exp()

    def select_veh_stochastic(self, state, mask_veh):
        batch_size, num_veh = state.capacity.size()

        vehicles = torch.arange(0, num_veh, dtype=torch.int64, device=mask_veh.device)
        available_vehicles = torch.arange(0, batch_size, dtype=torch.int64, device=mask_veh.device)
        # print("available_vehicles:", vehicles[~mask_veh[0]])
        for i in range(batch_size):
            available_vehicles[i] = random.choice(vehicles[~mask_veh[i]])

        return available_vehicles, torch.zeros([1, num_veh], dtype=torch.float32, device=state.capacity.device)

    def _inner(self, input, embeddings, engine: SimulatorEngine):
        # input: [batch_size, veh_size + graph_size, node_dim]
        # embeddings: [batch_size, veh_size + graph_size, embed_dim]
        state = self.problem.make_state(input, engine)
        batch_size, num_veh = state.capacity.size()

        candidate_operations = []
        candidate_operations_probs = []
        selected_operation = []
        selected_operation_probs = []
        outputs_veh_probs = []
        selected_veh_probs = []
        invalid_veh = []

        # Compute graph_embed and its projection(fixed context)
        fixed = self._precompute(embeddings)  # embeddings, context_node_project(graph_embed)

        # Perform decoding steps
        i = 0

        veh_list = []

        start_time = datetime.datetime.now()
        while num_veh > 0: # 修改此处的条件
            mask_veh = state.get_mask_veh(invalid_veh)

            if mask_veh.all():
                break

            if engine.strategy.opts.vehicle_strategy == 'decoder':
                veh, log_p_veh = self.select_veh(input, state, embeddings, mask_veh)
            elif engine.strategy.opts.vehicle_strategy == 'random':
                veh, log_p_veh = self.select_veh_stochastic(state, mask_veh)
            else:
                raise ValueError("strategy must be subset of vehicle strategies contains")

            veh_list.append(veh.tolist())
            outputs_veh_probs.append(log_p_veh)

            # final_avail_pos: 当前可选的操作 是否有为空的情况？
            log_p, final_avail_pos = self._get_log_p(fixed, state, veh, engine)
            if min(final_avail_pos[0].shape) == 0: # 当前车辆的路线无可选操作
                invalid_veh.append(veh)
                veh_list.pop()
                continue
            candidate_operations.append(final_avail_pos)
            candidate_operations_probs.append(log_p)

            # Select the next action in the final_avail_pos, result (batch_size, num_veh) long
            selected, selected_probs = self._select_node(log_p, final_avail_pos, state, veh)

            state = state.update(selected, veh, engine)

            # 每个步骤中每个batch对应的操作
            selected_operation.append(selected)
            selected_operation_probs.append(selected_probs)

            i += 1

            # self._display_select_action(i, state, selected, veh)

        end_time = datetime.datetime.now()
        # print("Generate routes strategy's time cost:", end_time - start_time)
        # print(len(veh_list))
        if(len(veh_list)==0):
            veh_list.append([])
        if(len(outputs_veh_probs)==0):
            outputs_veh_probs.append(torch.tensor([0 for i in range(0,num_veh)],dtype=torch.bool,device=state.capacity.device))
        veh_list = torch.tensor(veh_list).transpose(0, 1)
        # 每个步骤对应batch的可选位置
        return candidate_operations, candidate_operations_probs, selected_operation, selected_operation_probs, veh_list, \
               state.cur_route, state.cur_route_time_cost, torch.stack(outputs_veh_probs, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        # print('input', input)

        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi, veh_list, tour_1, tour_2, tour_3: self.problem.get_costs(input[0], self.obj, pi, veh_list, tour_1, tour_2, tour_3),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, final_avail_pos, state, veh):  # probs, mask: [batch_size, graph_size]
        selected = []
        selected_probs = []
        batch_size, num_veh = state.capacity.size()

        if self.decode_type == "greedy":
            for i in range(batch_size):
                _, selected_pos = probs[i].exp()[0, :].max(0)
                selected.append(final_avail_pos[i][selected_pos])
                selected_probs.append(probs[i][:, selected_pos].squeeze(0))

        elif self.decode_type == "sampling":
            for i in range(batch_size):
                selected_pos = probs[i].exp()[0, :].multinomial(1)
                selected.append((final_avail_pos[i][selected_pos]).squeeze(0))
                selected_probs.append(probs[i][:, selected_pos].squeeze(0))

        else:
            assert False, "Unknown decode type"

        return selected, selected_probs

    def _precompute(self, embeddings, num_steps=1):
        # embeddings: [batch_size, graph_size+1, embed_dim]

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)  # [batch_size, embed_dim]
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]  # linear(graph_embed)

        node_embeddings = embeddings[:, :, :]
        # The projection of the node embeddings for the attention is calculated once up front
        # glimpse_key_fixed size is torch.Size([batch_size, 1, graph_size+1, embed_dim])
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(node_embeddings[:, None, :, :]).chunk(3,
                                                                          dim=-1)  # split tensor to three parts in dimension 1

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (  # make multihead
            self._make_heads(glimpse_key_fixed, num_steps),  # (n_heads, batch_size, num_steps, graph_size, embed_dim)
            self._make_heads(glimpse_val_fixed, num_steps),  # (n_heads, batch_size, num_steps, graph_size, embed_dim)
            logit_key_fixed.contiguous()  # [batch_size, 1, graph_size+1, embed_dim]
        )
        return AttentionModelFixed(node_embeddings, fixed_context)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, veh, engine, normalize=True):
        # fixed: node_embeddings(embeddings), context_node_project(graph_embed), glimpse_key, glimpse_val, logits_key
        # Compute query = context node embedding, 相同维度数字相加
        # fixed.context_node_projected (graph_embedding): (batch_size, 1, embed_dim), query: [batch_size, num_veh, embed_dim]

        # self.project_step_context()
        route_embeddings = self._get_parallel_step_context(fixed.node_embeddings, state, veh)
        batch_size, _ = state.demand.size()

        # query = fixed.context_node_projected + route_embeddings

        # Compute the mask
        final_avail_pos = state.get_avail_pos(veh, engine)  # [batch_size, 1, graph_size]

        if min(final_avail_pos[0].shape) == 0: # 无可插入的位置
            return None, final_avail_pos

        if engine.strategy.opts.node_strategy == 'decoder': # 结点选择解码器

            # Compute keys and values for the nodes
            glimpse_K, glimpse_V, logit_K = self._insert_route_embeddings(fixed, state, veh, final_avail_pos)

            # Compute logits (unnormalized log_p)  log_p:[batch_size, num_veh, graph_size], glimpse:[batch_size, num_veh, embed_dim]
            logs_p = self._one_to_many_logits(route_embeddings, glimpse_K, glimpse_V, logit_K, veh)

            for log_p in logs_p:
                if normalize:
                    log_p = F.log_softmax(log_p / self.temp, dim=-1)

        elif engine.strategy.opts.node_strategy == 'no decoder': # 选择具有最小插入成本的结点
            final_avail_pos = self.select_min_cost_action(final_avail_pos, state, veh, engine)
            logs_p = []
            for _ in range(0, batch_size):
                logs_p.append(torch.zeros([1,1], dtype=torch.float32, device=state.capacity.device))
        else:
            pass

        return logs_p, final_avail_pos

    def select_min_cost_action(self, final_avail_pos, state, veh, engine):
        batch_size, graph_size = state.demand.size()
        # batch_size = len(final_avail_pos)
        num_veh = state.capacity.size(-1)

        for i in range(0, batch_size):
            route = state.cur_route[i][veh[i]].copy()  # 当前车辆的路线
            min_index = -1
            min_cost = float('inf')
            for ii in range(0, final_avail_pos[i].size(0)):
                update_route = route.copy()
                insert_node, insert_pos = final_avail_pos[i][ii][1].item(), final_avail_pos[i][ii][2].item()
                update_route.insert(insert_pos + 1, insert_node + num_veh)

                cur_node = update_route[0]
                ts = engine.timestamp
                a_cost = 0
                for iii in range(1, len(update_route)):
                    next_node = update_route[iii]
                    source_node, target_node = cur_node, next_node
                    if cur_node < num_veh:
                        source_node, target_node = get_vh_source_and_target(engine, cur_node, next_node - num_veh)
                    else:
                        source_node, target_node = get_source_and_target(engine, cur_node - num_veh,
                                                                         next_node - num_veh)
                    _, cost = engine.vehicle_manager.shortest_travel_path_cost(source_node, target_node, ts)
                    if cost is None:
                        flag = True
                        break
                    a_cost += cost

                    cur_node = next_node
                    ts += pd.Timedelta(cost, unit='s')
                if a_cost < min_cost:
                    min_cost = a_cost
                    min_index = ii

            if min_index != -1:
                final_avail_pos[i] = final_avail_pos[i][min_index, :][None,:]   # 返回具有最小成本的
            else:
                engine.strategy.logger.error("没有可选的动作！")

        return final_avail_pos

    def _get_parallel_step_context(self, embeddings, state, veh, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        cur_route = state.get_cur_route()
        batch_size, num_veh = state.capacity.size()
        num_steps = 1

        route_index = []
        for i in range(0, veh.size(-1)):
            route = torch.tensor(cur_route[i][veh[i]], device=veh.device)
            route_index.append(route.contiguous().view(route.size(-1), 1).expand(-1, embeddings.size(-1)))

        route_embeddings = []
        # route_embeddings = torch.empty((batch_size, 1, embeddings.size(-1)), device=veh.device)
        for index, value in enumerate(route_index):
            # route_embeddings[index, :, :] = torch.gather(embeddings[index, :, :], 0, value).mean(0)
            # route_embeddings[index, :, :] = torch.gather(embeddings[index, :, :], 0, value)
            route_length = value.size(0)
            if route_length < 16:
                padding = torch.zeros([16-route_length, embeddings.size(-1)], dtype=value.dtype, device=value.device)
                route_embeddings.append(
                    torch.cat(
                        (
                            torch.gather(embeddings[index, :, :], 0, value),
                            padding
                        ),
                        0
                    )
                )
            else:
                route_embeddings.append(torch.gather(embeddings[index, :, :], 0, value))
            route_embeddings[index] = self.project_route_embeddings(route_embeddings[index].view(-1)).unsqueeze(0)
        return route_embeddings

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, veh):
        batch_size = len(query)
        num_step = glimpse_K[0].size(1)
        embed_dim = logit_K[0].size(-1)
        key_size = val_size = embed_dim // self.n_heads  # query and K both have key_size
        logits = []

        for i in range(batch_size):
            # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, num_step, 1, route_length, key_size)
            glimpse_Q = query[i].view(num_step, self.n_heads, 1, key_size).permute(1, 0, 2, 3)

            # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_step, 1, graph_size)
            # glimpse_K (n_heads, num_step, graph_size, route_length + 1, key_size)
            compatibility = torch.matmul(glimpse_Q, glimpse_K[i].transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

            # Batch matrix multiplication to compute heads (n_heads, batch_size, num_step, 1, val_size)
            heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V[i])

            # Project to get glimpse/updated context node embedding (batch_size, num_step, 1, embedding_dim)
            glimpse = self.project_out(
                heads.permute(1, 2, 0, 3).contiguous().view(num_step, 1, self.n_heads * val_size))

            # Now projecting the glimpse is not needed since this can be absorbed into project_out
            final_Q = glimpse

            # logits_K, (batch_size, 1, graph_size, embed_dim)
            # Batch matrix multiplication to compute logits (batch_size, num_step, graph_size)
            logit = torch.matmul(final_Q, logit_K[i].transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

            # From the logits compute the probabilities by clipping, masking and softmax
            if self.tanh_clipping > 0:  # 10
                logit = torch.tanh(logit) * self.tanh_clipping

            logits.append(logit)
        # return logits, glimpse.squeeze(-2)  # glimpse[batch_size, num_veh, embed_dim]
        return logits

    def _insert_route_embeddings(self, fixed, state, veh, final_avail_pos):
        batch_size, num_veh = state.capacity.size()
        embeddings = fixed.node_embeddings

        # compute key and value
        insert_route_index = [[] for i in range(veh.size(-1))]
        for i in range(0, veh.size(-1)):
            for k in range(0, final_avail_pos[i].size(0)):
                plan_route = state.cur_route[i][veh[i]].copy()  # 需要向嵌套列表中插入新的结点 计算所有可能的路线 不能修改原来的计划路线
                insert_index = final_avail_pos[i][k, 2].item() + 1
                insert_node = final_avail_pos[i][k, 1].item() + num_veh

                plan_route.insert(insert_index, insert_node)
                insert_route = torch.tensor(plan_route, device=veh.device)
                insert_route_index[i].append(
                    insert_route.contiguous().view(insert_route.size(-1), 1).expand(-1, embeddings.size(-1)))

        insert_route_embeddings = [[] for _ in range(veh.size(-1))]
        for i in range(0, veh.size(-1)):
            for value in insert_route_index[i]:
                route_length = value.size(0)
                if route_length < 16:
                    padding = torch.zeros([16-route_length, embeddings.size(-1)], dtype=value.dtype, device=value.device)
                    insert_route_embeddings[i].append(
                        torch.cat(
                            (
                                torch.gather(embeddings[i, :, :], 0, value),
                                padding
                            ),
                            0
                        )
                    )
                else:
                    insert_route_embeddings[i].append(torch.gather(embeddings[i, :, :], 0, value))
            try:
                insert_route_embeddings[i] = torch.stack(insert_route_embeddings[i], 0)
            except RuntimeError:
                print("stack expects a non-empty TensorList")
            route_size = insert_route_embeddings[i].size(0)
            insert_route_embeddings[i] = self.project_insert_route_embeddings(insert_route_embeddings[i].view(route_size, -1))

        glimpse_key = []
        glimpse_val = []
        logit_key = []

        for i in range(batch_size):
            key, val, _key = (self.project_node_embeddings(insert_route_embeddings[i][None, :, :]).chunk(3, dim=-1))
            glimpse_key.append(self._make_heads(key)) # (n_heads, num_steps, graph_size, route_length, embed_dim)
            glimpse_val.append(self._make_heads(val))
            logit_key.append(_key.contiguous())

        return glimpse_key, glimpse_val, logit_key

    def _make_heads(self, v, num_steps=1):  # v: [ 1, graph_size+1, embed_dim]
        assert num_steps is None or v.size(0) == 1 or v.size(0) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), self.n_heads, -1)
                .expand(v.size(0) if num_steps is None else num_steps, v.size(1) , self.n_heads, -1)
                .permute(2, 0, 1, 3)  # (n_heads, batch_size, num_steps, graph_size, embed_dim)
        )

        # assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        #
        # return (
        #     v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
        #         .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
        #         .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, embed_dim)
        # )

    def _display_select_action(self, step, state, selected, veh):
        # display result
        vehicles_code = veh.item()
        graph_size = state.demand.size(-1) // 2
        num_veh = state.capacity.size(-1)
        request_code = ''
        if selected[0][1] < graph_size:
            request_code = 'o' + str(selected[0][1].item())
        else:
            request_code = 'd' + str((selected[0][1] - graph_size).item())

        route = []
        for r in state.cur_route[0][veh[0]]:
            if r < num_veh:
                route.append('v' + str(r))
            elif r < num_veh + graph_size:
                route.append('o' + str(r - num_veh))
            else:
                route.append('d' + str(r - num_veh - graph_size))
        print("step:", step, " 选择的车辆: ", vehicles_code, " 选择的结点:", request_code, " 插入位置:", selected[0][2].item(), " 插入后的路线:", \
              " --> ".join(route))


