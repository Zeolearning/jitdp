import queue
import networkx as nx
from .util import CONSTANTS


class Slicing:

    def __init__(self):
        pass

    def forward_dependency_slicing(self, node, graph: nx.MultiDiGraph,line_set=None):
        line_ctx = dict()
        visited = set()
        n_statement = set()
        q = queue.Queue()
        q.put(node)
        visited.add(node)
        ddg_hop=CONSTANTS.ddg_max_hop
        cdg_hop=CONSTANTS.cdg_max_hop
        def cdg_view(v, u, t):
            return t == 'CDG'

        def ddg_view(v, u, t):
            return t == 'DDG'

        cdg = nx.subgraph_view(graph, filter_edge=cdg_view)
        ddg = nx.subgraph_view(graph, filter_edge=ddg_view)


        while ddg_hop>0 and not q.empty():
            temp_q=queue.Queue()
            while not q.empty():
                curr_v = q.get()
                #向上寻找ddg hop内数据依赖
                for u in ddg.predecessors(curr_v):
                    if u not in visited:
                        temp_q.put(u)
                        visited.add(u)
            ddg_hop-=1
            q=temp_q

        q=queue.Queue()
        cdg_visited=set()
        for n in visited:
            q.put(n)
            cdg_visited.add(n)
        while(cdg_hop>0 and not q.empty()):
            temp_q=queue.Queue()
            while not q.empty():
                curr_v = q.get()
                #向上寻找控制依赖
                for u in cdg.predecessors(curr_v):
                    if u not in cdg_visited:
                        temp_q.put(u)
                        cdg_visited.add(u)
                        visited.add(u)
            cdg_hop-=1
            q=temp_q
        
        self.extract_lines(visited, graph, line_ctx, n_statement)

        line_list = list(line_ctx.keys())
        line_list.sort()
        filter_ctx = []
        ctx=[]
        for i in line_list:
            if i not in line_set:
                ctx.append(line_ctx[i])
                filter_ctx.append({i: line_ctx[i]})
                line_set.append(i)
        return "".join(ctx), filter_ctx, line_list, n_statement
    
    def backward_dependency_slicing(self, node, graph: nx.MultiDiGraph,line_set=None):
        line_ctx = dict()
        visited = set()
        n_statement = set()
        q = queue.Queue()
        q.put(node)
        visited.add(node)
        ddg_hop=CONSTANTS.ddg_max_hop
        cdg_hop=CONSTANTS.cdg_max_hop
        def cdg_view(v, u, t):
            return t == 'CDG'

        def ddg_view(v, u, t):
            return t == 'DDG'

        cdg = nx.subgraph_view(graph, filter_edge=cdg_view)
        ddg = nx.subgraph_view(graph, filter_edge=ddg_view)

        if "declaration" not in graph.nodes[node]['nodeType']:
            for x in cdg.successors(node):
                if x not in visited:
                    q.put(x)
                    visited.add(x)
                
        while ddg_hop>0 and not q.empty():
            temp_q=queue.Queue()
            while not q.empty():
                curr_v = q.get()
                #向下寻找ddg hop内数据依赖
                for u in ddg.successors(curr_v):
                    if u not in visited:
                        temp_q.put(u)
                        visited.add(u)
            ddg_hop-=1
            q=temp_q

        q=queue.Queue()
        cdg_visited=set()
        for n in visited:
            q.put(n)
            cdg_visited.add(n)
        while(cdg_hop>0 and not q.empty()):
            temp_q=queue.Queue()
            while not q.empty():
                curr_v = q.get()
                #向上寻找控制依赖
                for u in cdg.predecessors(curr_v):
                    if u not in cdg_visited:
                        temp_q.put(u)
                        cdg_visited.add(u)
                        visited.add(u)
            cdg_hop-=1
            q=temp_q
        self.extract_lines(visited, graph, line_ctx, n_statement)

        # n_statement.remove(node)
        # start_line = graph.nodes[node]['startRow']
        # end_line = graph.nodes[node]['endRow']
        # for i in range(start_line, end_line + 1):
        #     line_ctx.pop(i)

        line_list = list(line_ctx.keys())
        line_list.sort()
        filter_ctx = []
        ctx=[]
        for i in line_list:
            if i not in line_set:
                ctx.append(line_ctx[i])
                filter_ctx.append({i: line_ctx[i]})
                line_set.append(i)


        return "".join(ctx), filter_ctx, line_list, n_statement


    def extract_lines(self,p_set, graph, line_ctx, n_statement):
        for p in p_set:
            start_line = graph.nodes[p]['startRow']
            end_line = graph.nodes[p]['endRow']
            for i in range(start_line, end_line + 1):
                if i not in line_ctx:
                    line_ctx[i] = graph.nodes[p]['sourceLines'][i - start_line]
            n_statement.add(p)
