from __future__ import division
from collections import defaultdict
from heapq import heapify, heappush, heappop
from timeit import default_timer
from sys import stdout, version_info
from multiprocessing import Pool
from tqdm import tqdm
from warnings import warn
from math import pi, sqrt, atan, acos

INF = 10000
CCW = 1
CW = -1
COLLINEAR = 0
"""Due to floating point representation error, some functions need to
   truncate floating point numbers to a certain tolerance."""
COLIN_TOLERANCE = 10
T = 10**COLIN_TOLERANCE
T2 = 10.0**COLIN_TOLERANCE

try:
    dict.iteritems
except AttributeError:
    # Python 3
    def iteritems(d):
        return iter(d.items())

class Point(object):
    __slots__ = ('x', 'y', 'polygon_id')

    def __init__(self, x, y, polygon_id=-1):
        self.x = float(x)
        self.y = float(y)
        self.polygon_id = polygon_id

    def __eq__(self, point):
        return point and self.x == point.x and self.y == point.y

    def __ne__(self, point):
        return not self.__eq__(point)

    def __lt__(self, point):
        """ This is only needed for shortest path calculations where heapq is
            used. When there are two points of equal distance, heapq will
            instead evaluate the Points, which doesnt work in Python 3 and
            throw a TypeError."""
        return hash(self) < hash(point)

    def __str__(self):
        return "(%.2f, %.2f)" % (self.x, self.y)

    def __hash__(self):
        return self.x.__hash__() ^ self.y.__hash__()

    def __repr__(self):
        return "Point(%.2f, %.2f)" % (self.x, self.y)


class Edge(object):
    __slots__ = ('p1', 'p2')

    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2

    def get_adjacent(self, point):
        if point == self.p1:
            return self.p2
        return self.p1

    def __contains__(self, point):
        return self.p1 == point or self.p2 == point

    def __eq__(self, edge):
        if self.p1 == edge.p1 and self.p2 == edge.p2:
            return True
        if self.p1 == edge.p2 and self.p2 == edge.p1:
            return True
        return False

    def __ne__(self, edge):
        return not self.__eq__(edge)

    def __str__(self):
        return "({}, {})".format(self.p1, self.p2)

    def __repr__(self):
        return "Edge({!r}, {!r})".format(self.p1, self.p2)

    def __hash__(self):
        return self.p1.__hash__() ^ self.p2.__hash__()


class Graph(object):
    """
    A Graph is represented by a dict where the keys are Points in the Graph
    and the dict values are sets containing Edges incident on each Point.
    A separate set *edges* contains all Edges in the graph.
    The input must be a list of polygons, where each polygon is a list of
    in-order (clockwise or counter clockwise) Points. If only one polygon,
    it must still be a list in a list, i.e. [[Point(0,0), Point(2,0),
    Point(2,1)]].
    *polygons* dictionary: key is a integer polygon ID and values are the
    edges that make up the polygon. Note only polygons with 3 or more Points
    will be classified as a polygon. Non-polygons like just one Point will be
    given a polygon ID of -1 and not maintained in the dict.
    """

    def __init__(self, polygons):
        self.graph = defaultdict(set)
        self.edges = set()
        self.polygons = defaultdict(set)
        pid = 0
        for polygon in polygons:
            if polygon[0] == polygon[-1] and len(polygon) > 1:
                polygon.pop()
            for i, point in enumerate(polygon):
                sibling_point = polygon[(i + 1) % len(polygon)]
                edge = Edge(point, sibling_point)
                if len(polygon) > 2:
                    point.polygon_id = pid
                    sibling_point.polygon_id = pid
                    self.polygons[pid].add(edge)
                self.add_edge(edge)
            if len(polygon) > 2:
                pid += 1

    def get_adjacent_points(self, point):
        return [edge.get_adjacent(point) for edge in self[point]]

    def get_points(self):
        return list(self.graph)

    def get_edges(self):
        return self.edges

    def add_edge(self, edge):
        self.graph[edge.p1].add(edge)
        self.graph[edge.p2].add(edge)
        self.edges.add(edge)

    def __contains__(self, item):
        if isinstance(item, Point):
            return item in self.graph
        if isinstance(item, Edge):
            return item in self.edges
        return False

    def __getitem__(self, point):
        if point in self.graph:
            return self.graph[point]
        return set()

    def __str__(self):
        res = ""
        for point in self.graph:
            res += "\n" + str(point) + ": "
            for edge in self.graph[point]:
                res += str(edge)
        return res

    def __repr__(self):
        return self.__str__()


def dijkstra(graph, origin, destination, add_to_visgraph):
    D = {}
    P = {}
    Q = priority_dict()
    Q[origin] = 0

    for v in Q:
        D[v] = Q[v]
        if v == destination: break

        edges = graph[v]
        if add_to_visgraph != None and len(add_to_visgraph[v]) > 0:
            edges = add_to_visgraph[v] | graph[v]
        for e in edges:
            w = e.get_adjacent(v)
            elength = D[v] + edge_distance(v, w)
            if w in D:
                if elength < D[w]:
                    raise ValueError
            elif w not in Q or elength < Q[w]:
                Q[w] = elength
                P[w] = v
    return (D, P)


def shortest_path(graph, origin, destination, add_to_visgraph=None):
    D, P = dijkstra(graph, origin, destination, add_to_visgraph)
    path = []
    while 1:
        path.append(destination)
        if destination == origin: break
        destination = P[destination]
    path.reverse()
    return path


class priority_dict(dict):
    """Dictionary that can be used as a priority queue.
    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is that priorities
    of items can be efficiently updated (amortized O(1)) using code as
    'thedict[item] = new_priority.'
    Note that this is a modified version of
    https://gist.github.com/matteodellamico/4451520 where sorted_iter() has
    been replaced with the destructive sorted iterator __iter__ from
    https://gist.github.com/anonymous/4435950
    """
    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in iteritems(self)]
        heapify(self._heap)

    def smallest(self):
        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        super(priority_dict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def __iter__(self):
        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]
        return iterfn()


class VisGraph(object):

    def __init__(self):
        self.graph = None
        self.visgraph = None

    def load(self, filename):
        """Load obstacle graph and visibility graph. """
        with open(filename, 'rb') as load:
            self.graph, self.visgraph = pickle.load(load)

    def save(self, filename):
        """Save obstacle graph and visibility graph. """
        with open(filename, 'wb') as output:
            pickle.dump((self.graph, self.visgraph), output, -1)

    def build(self, input, status=True): 
        """Build visibility graph based on a list of polygons.
        The input must be a list of polygons, where each polygon is a list of
        in-order (clockwise or counter clockwise) Points. It only one polygon,
        it must still be a list in a list, i.e. [[Point(0,0), Point(2,0),
        Point(2,1)]].
        Take advantage of processors with multiple cores by setting workers to
        the number of subprocesses you want. Defaults to 1, i.e. no subprocess
        will be started.
        Set status=False to turn off the statusbar when building.
        """

        self.graph = Graph(input)
        self.visgraph = Graph([])

        points = self.graph.get_points()
        batch_size = 10 

        for batch in tqdm([points[i:i + batch_size]
                          for i in range(0, len(points), batch_size)],
                          disable=not status):
            for edge in _vis_graph(self.graph, batch):
                self.visgraph.add_edge(edge)

    def find_visible(self, point):
        """Find vertices visible from point."""

        return visible_vertices(point, self.graph)

    def update(self, points, origin=None, destination=None):
        """Update visgraph by checking visibility of Points in list points."""

        for p in points:
            for v in visible_vertices(p, self.graph, origin=origin,
                                      destination=destination):
                self.visgraph.add_edge(Edge(p, v))

    def shortest_path(self, origin, destination):
        """Find and return shortest path between origin and destination.
        Will return in-order list of Points of the shortest path found. If
        origin or destination are not in the visibility graph, their respective
        visibility edges will be found, but only kept temporarily for finding
        the shortest path. 
        """

        origin_exists = origin in self.visgraph
        dest_exists = destination in self.visgraph
        if origin_exists and dest_exists:
            return shortest_path(self.visgraph, origin, destination)
        orgn = None if origin_exists else origin
        dest = None if dest_exists else destination
        add_to_visg = Graph([])
        if not origin_exists:
            for v in visible_vertices(origin, self.graph, destination=dest):
                add_to_visg.add_edge(Edge(origin, v))
        if not dest_exists:
            for v in visible_vertices(destination, self.graph, origin=orgn):
                add_to_visg.add_edge(Edge(destination, v))
        return shortest_path(self.visgraph, origin, destination, add_to_visg)

    def point_in_polygon(self, point):
        """Return polygon_id if point in a polygon, -1 otherwise."""

        return point_in_polygon(point, self.graph)

    def closest_point(self, point, polygon_id, length=0.001):
        """Return closest Point outside polygon from point.
        Note method assumes point is inside the polygon, no check is
        performed.
        """

        return closest_point(point, self.graph, polygon_id, length)


def _vis_graph_wrapper(args):
    try:
        return _vis_graph(*args)
    except KeyboardInterrupt:
        pass

def _vis_graph(graph, points):
    visible_edges = []
    for p1 in points:
        for p2 in visible_vertices(p1, graph, scan='half'):
            visible_edges.append(Edge(p1, p2))
    return visible_edges

def visible_vertices(point, graph, origin=None, destination=None, scan='full'):
    """Returns list of Points in graph visible by point.
    If origin and/or destination Points are given, these will also be checked
    for visibility. scan 'full' will check for visibility against all points in
    graph, 'half' will check for visibility against half the points. This saves
    running time when building a complete visibility graph, as the points
    that are not checked will eventually be 'point'.
    """
    edges = graph.get_edges()
    points = graph.get_points()
    if origin: points.append(origin)
    if destination: points.append(destination)
    points.sort(key=lambda p: (angle(point, p), edge_distance(point, p)))

    # Initialize open_edges with any intersecting edges on the half line from
    # point along the positive x-axis
    open_edges = OpenEdges()
    point_inf = Point(INF, point.y)
    for edge in edges:
        if point in edge: continue
        if edge_intersect(point, point_inf, edge):
            if on_segment(point, edge.p1, point_inf): continue
            if on_segment(point, edge.p2, point_inf): continue
            open_edges.insert(point, point_inf, edge)

    visible = []
    prev = None
    prev_visible = None
    for p in points:
        if p == point: continue
        if scan == 'half' and angle(point, p) > pi: break

        # Update open_edges - remove clock wise edges incident on p
        if open_edges:
            for edge in graph[p]:
                if ccw(point, p, edge.get_adjacent(p)) == CW:
                    open_edges.delete(point, p, edge)

        # Check if p is visible from point
        is_visible = False
        # ...Non-collinear points
        if prev is None or ccw(point, prev, p) != COLLINEAR or not on_segment(point, prev, p):
            if len(open_edges) == 0:
                is_visible = True
            elif not edge_intersect(point, p, open_edges.smallest()):
                is_visible = True
        # ...For collinear points, if previous point was not visible, p is not
        elif not prev_visible:
            is_visible = False
        # ...For collinear points, if previous point was visible, need to check
        # that the edge from prev to p does not intersect any open edge.
        else:
            is_visible = True
            for edge in open_edges:
                if prev not in edge and edge_intersect(prev, p, edge):
                    is_visible = False
                    break
            if is_visible and edge_in_polygon(prev, p, graph):
                    is_visible = False

        # Check if the visible edge is interior to its polygon
        if is_visible and p not in graph.get_adjacent_points(point):
            is_visible = not edge_in_polygon(point, p, graph)

        if is_visible: visible.append(p)

        # Update open_edges - Add counter clock wise edges incident on p
        for edge in graph[p]:
            if (point not in edge) and ccw(point, p, edge.get_adjacent(p)) == CCW:
                open_edges.insert(point, p, edge)

        prev = p
        prev_visible = is_visible
    return visible


def polygon_crossing(p1, poly_edges):
    """Returns True if Point p1 is internal to the polygon. The polygon is
    defined by the Edges in poly_edges. Uses crossings algorithm and takes into
    account edges that are collinear to p1."""
    p2 = Point(INF, p1.y)
    intersect_count = 0
    for edge in poly_edges:
        if p1.y < edge.p1.y and p1.y < edge.p2.y: continue
        if p1.y > edge.p1.y and p1.y > edge.p2.y: continue
        if p1.x > edge.p1.x and p1.x > edge.p2.x: continue
        # Deal with points collinear to p1
        edge_p1_collinear = (ccw(p1, edge.p1, p2) == COLLINEAR)
        edge_p2_collinear = (ccw(p1, edge.p2, p2) == COLLINEAR)
        if edge_p1_collinear and edge_p2_collinear: continue
        if edge_p1_collinear or edge_p2_collinear:
            collinear_point = edge.p1 if edge_p1_collinear else edge.p2
            if edge.get_adjacent(collinear_point).y > p1.y:
                intersect_count += 1
        elif edge_intersect(p1, p2, edge):
            intersect_count += 1
    if intersect_count % 2 == 0:
        return False
    return True


def edge_in_polygon(p1, p2, graph):
    """Return true if the edge from p1 to p2 is interior to any polygon
    in graph."""
    if p1.polygon_id != p2.polygon_id:
        return False
    if p1.polygon_id == -1 or p2.polygon_id == -1:
        return False
    mid_point = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
    return polygon_crossing(mid_point, graph.polygons[p1.polygon_id])


def point_in_polygon(p, graph):
    """Return true if the point p is interior to any polygon in graph."""
    for polygon in graph.polygons:
        if polygon_crossing(p, graph.polygons[polygon]):
            return polygon
    return -1


def unit_vector(c, p):
    magnitude = edge_distance(c, p)
    return Point((p.x - c.x) / magnitude, (p.y - c.y) / magnitude)


def closest_point(p, graph, polygon_id, length=0.001):
    """Assumes p is interior to the polygon with polygon_id. Returns the
    closest point c outside the polygon to p, where the distance from c to
    the intersect point from p to the edge of the polygon is length."""
    polygon_edges = graph.polygons[polygon_id]
    close_point = None
    close_edge = None
    close_dist = None
    # Finds point closest to p, but on a edge of the polygon.
    # Solution from http://stackoverflow.com/a/6177788/4896361
    for i, e in enumerate(polygon_edges):
        num = ((p.x-e.p1.x)*(e.p2.x-e.p1.x) + (p.y-e.p1.y)*(e.p2.y-e.p1.y))
        denom = ((e.p2.x - e.p1.x)**2 + (e.p2.y - e.p1.y)**2)
        u = num/denom
        pu = Point(e.p1.x + u*(e.p2.x - e.p1.x), e.p1.y + u*(e.p2.y- e.p1.y))
        pc = pu
        if u < 0:
            pc = e.p1
        elif u > 1:
            pc = e.p2
        d = edge_distance(p, pc)
        if i == 0 or d < close_dist:
            close_dist = d
            close_point = pc
            close_edge = e

    # Extend the newly found point so it is outside the polygon by `length`.
    if close_point in close_edge:
        c = close_edge.p1 if close_point == close_edge.p1 else close_edge.p2
        edges = list(graph[c])
        v1 = unit_vector(c, edges[0].get_adjacent(c))
        v2 = unit_vector(c, edges[1].get_adjacent(c))
        vsum = unit_vector(Point(0, 0), Point(v1.x + v2.x, v1.y + v2.y))
        close1 = Point(c.x + (vsum.x * length), c.y + (vsum.y * length))
        close2 = Point(c.x - (vsum.x * length), c.y - (vsum.y * length))
        if point_in_polygon(close1, graph) == -1:
            return close1
        return close2
    else:
        v = unit_vector(p, close_point)
        return Point(close_point.x + v.x*length, close_point.y + v.y*length)


def edge_distance(p1, p2):
    """Return the Euclidean distance between two Points."""
    return sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)


def intersect_point(p1, p2, edge):
    """Return intersect Point where the edge from p1, p2 intersects edge"""
    if p1 in edge: return p1
    if p2 in edge: return p2
    if edge.p1.x == edge.p2.x:
        if p1.x == p2.x:
            return None
        pslope = (p1.y - p2.y) / (p1.x - p2.x)
        intersect_x = edge.p1.x
        intersect_y = pslope * (intersect_x - p1.x) + p1.y
        return Point(intersect_x, intersect_y)

    if p1.x == p2.x:
        eslope = (edge.p1.y - edge.p2.y) / (edge.p1.x - edge.p2.x)
        intersect_x = p1.x
        intersect_y = eslope * (intersect_x - edge.p1.x) + edge.p1.y
        return Point(intersect_x, intersect_y)

    pslope = (p1.y - p2.y) / (p1.x - p2.x)
    eslope = (edge.p1.y - edge.p2.y) / (edge.p1.x - edge.p2.x)
    if eslope == pslope:
        return None
    intersect_x = (eslope * edge.p1.x - pslope * p1.x + p1.y - edge.p1.y) / (eslope - pslope)
    intersect_y = eslope * (intersect_x - edge.p1.x) + edge.p1.y
    return Point(intersect_x, intersect_y)


def point_edge_distance(p1, p2, edge):
    """Return the Eucledian distance from p1 to intersect point with edge.
    Assumes the line going from p1 to p2 intersects edge before reaching p2."""
    ip = intersect_point(p1, p2, edge)
    if ip is not None:
        return edge_distance(p1, ip)
    return 0


def angle(center, point):
    """Return the angle (radian) of point from center of the radian circle.
     ------p
     |   /
     |  /
    c|a/
    """
    dx = point.x - center.x
    dy = point.y - center.y
    if dx == 0:
        if dy < 0:
            return pi * 3 / 2
        return pi / 2
    if dy == 0:
        if dx < 0:
            return pi
        return 0
    if dx < 0:
        return pi + atan(dy / dx)
    if dy < 0:
        return 2 * pi + atan(dy / dx)
    return atan(dy / dx)


def angle2(point_a, point_b, point_c):
    """Return angle B (radian) between point_b and point_c.
           c
         /  \
       /    B\
      a-------b
    """
    a = (point_c.x - point_b.x)**2 + (point_c.y - point_b.y)**2
    b = (point_c.x - point_a.x)**2 + (point_c.y - point_a.y)**2
    c = (point_b.x - point_a.x)**2 + (point_b.y - point_a.y)**2
    cos_value = (a + c - b) / (2 * sqrt(a) * sqrt(c))
    return acos(int(cos_value*T)/T2)


def ccw(A, B, C):
    """Return 1 if counter clockwise, -1 if clock wise, 0 if collinear """
    #  Rounding this way is faster than calling round()
    area = int(((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x))*T)/T2
    if area > 0: return 1
    if area < 0: return -1
    return 0


def on_segment(p, q, r):
    """Given three colinear points p, q, r, the function checks if point q
    lies on line segment 'pr'."""
    if (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)):
        if (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y)):
            return True
    return False


def edge_intersect(p1, q1, edge):
    """Return True if edge from A, B interects edge.
    http://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/"""
    p2 = edge.p1
    q2 = edge.p2
    o1 = ccw(p1, q1, p2)
    o2 = ccw(p1, q1, q2)
    o3 = ccw(p2, q2, p1)
    o4 = ccw(p2, q2, q1)

    # General case
    if (o1 != o2 and o3 != o4):
        return True
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if o1 == COLLINEAR and on_segment(p1, p2, q1):
        return True
    # p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if o2 == COLLINEAR and on_segment(p1, q2, q1):
        return True
    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if o3 == COLLINEAR and on_segment(p2, p1, q2):
        return True
    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if o4 == COLLINEAR and on_segment(p2, q1, q2):
        return True
    return False


class OpenEdges(object):
    def __init__(self):
        self._open_edges = []

    def insert(self, p1, p2, edge):
        self._open_edges.insert(self._index(p1, p2, edge), edge)

    def delete(self, p1, p2, edge):
        index = self._index(p1, p2, edge) - 1
        if self._open_edges[index] == edge:
            del self._open_edges[index]

    def smallest(self):
        return self._open_edges[0]

    def _less_than(self, p1, p2, edge1, edge2):
        """Return True if edge1 is smaller than edge2, False otherwise."""
        if edge1 == edge2:
            return False
        if not edge_intersect(p1, p2, edge2):
            return True
        edge1_dist = point_edge_distance(p1, p2, edge1)
        edge2_dist = point_edge_distance(p1, p2, edge2)
        if edge1_dist > edge2_dist:
            return False
        if edge1_dist < edge2_dist:
            return True
        # If the distance is equal, we need to compare on the edge angles.
        if edge1_dist == edge2_dist:
            if edge1.p1 in edge2:
                same_point = edge1.p1
            else:
                same_point = edge1.p2
            angle_edge1 = angle2(p1, p2, edge1.get_adjacent(same_point))
            angle_edge2 = angle2(p1, p2, edge2.get_adjacent(same_point))
            if angle_edge1 < angle_edge2:
                return True
            return False

    def _index(self, p1, p2, edge):
        lo = 0
        hi = len(self._open_edges)
        while lo < hi:
            mid = (lo+hi)//2
            if self._less_than(p1, p2, edge, self._open_edges[mid]):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def __len__(self):
        return len(self._open_edges)

    def __getitem__(self, index):
        return self._open_edges[index]

polys = [[Point(0.0,1.0), Point(3.0,1.0), Point(1.5,4.0)],
         [Point(4.0,4.0), Point(7.0,4.0), Point(5.5,8.0)]]
g = VisGraph()
g.build(polys)
shortest = g.shortest_path(Point(1.5,0.0), Point(4.0, 6.0))
print (shortest)