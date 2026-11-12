import numpy as np
from collections import deque
import heapq

DIRS = (
    ("ACTION1", -2, 0),
    ("ACTION2", 2, 0),
    ("ACTION3", 0, -2),
    ("ACTION4", 0, 2),
)

CR1 = np.array([
    [0,0,0,1,1,1,1,1],
    [0,0,0,1,0,0,0,1],
    [0,1,1,1,1,0,0,0],
    [1,1,1,1,0,0,0,0],
    [1,1,1,1,0,0,0,0],
    [0,1,1,1,1,0,0,0],
    [0,0,0,1,0,0,0,1],
    [0,0,0,1,1,1,1,1],
], dtype=bool)

CR2 = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1],
    [0,1,1,1,1,0,0,1],
    [1,1,1,1,0,0,0,0],
    [1,1,1,1,0,0,0,0],
    [0,1,1,1,1,0,0,1],
    [0,0,0,1,1,1,1,1],
    [0,0,0,0,0,0,0,0],
], dtype=bool)


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for yy, xx in zip(*np.nonzero(mask)):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(y, x)]
        pts = []
        while stack:
            py, px = stack.pop()
            pts.append((py, px))
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                ny, nx = py+dy, px+dx
                if (0 <= ny < h and 0 <= nx < w and
                        mask[ny,nx] and not seen[ny,nx]):
                    seen[ny,nx] = True
                    stack.append((ny,nx))
        ys, xs = zip(*pts)
        out.append((min(ys), min(xs), max(ys)+1, max(xs)+1, pts))
    return out


def get_cut(obs):
    candidates = []
    for x in range(20, 59):
        zeros = int(np.count_nonzero(obs[:63,x] == 0))
        black = int(np.count_nonzero(obs[:63,x+2] == 5))
        if zeros >= 8 and black >= 15:
            candidates.append((zeros + .15*black, x))
    if candidates:
        best = max(s for s,x in candidates)
        return min(x for s,x in candidates if s >= best-2)
    for x in range(20,59):
        if np.count_nonzero(obs[:63,x] == 5) > 35:
            return x
    return 38


def get_player(obs):
    y,x = np.nonzero(obs == 14)
    return int(y.min())*64 + int(x.min()) if len(y) else None


def get_goal(obs, cut):
    for y,x,y1,x1,pts in components(obs[:63,:cut] == 11):
        if y1-y <= 3 and x1-x <= 3:
            return y*64+x
    return None


def find_crane(a, cut):
    for y,x in zip(*np.nonzero(a[:63,:cut] == 8)):
        y,x = int(y)-3,int(x)
        if y < 0 or y+8 > 63 or x+8 > cut:
            continue
        patch = a[y:y+8,x:x+8] == 8
        if np.array_equal(patch,CR1):
            return y,x,False
        if np.array_equal(patch,CR2):
            return y,x,True
    return None


def terrain(a, cut):
    floor = np.isin(a,[1,2,6,7,8,10,12,13])
    block = np.zeros((64,64), dtype=bool)
    for color in (9,15):
        pixels = a == color
        adjacent = np.zeros_like(pixels)
        adjacent[1:] |= pixels[:-1]
        adjacent[:-1] |= pixels[1:]
        adjacent[:,1:] |= pixels[:,:-1]
        adjacent[:,:-1] |= pixels[:,1:]
        solid = pixels & adjacent
        floor |= solid
        block |= pixels & ~adjacent
    crane = find_crane(a,cut)
    if crane:
        y,x,attached = crane
        shape = CR2 if attached else CR1
        floor[y:y+8,x:x+8][shape] = False
        block[y:y+8,x:x+8] |= shape
    floor[:,cut:] = False
    floor[63,:] = False
    block[:,cut:] = True
    return floor,block


def legal_positions(floor, block):
    b = block.copy()
    b[:-1,:] |= block[1:,:]
    b[:,:-1] |= block[:,1:]
    b[:-1,:-1] |= block[1:,1:]
    result = floor & ~b
    result[63,:] = False
    result[:,63] = False
    return result.ravel()


def clean_observation(obs, mem, p):
    a = obs.copy()
    cut = mem["cut"]
    a[63,:] = 4
    a[:,cut:] = 4
    a[a == 3] = 4
    a[a == 11] = 2
    ys,xs = np.nonzero(a == 14)
    if not len(ys):
        return a
    y,x = int(ys.min()),int(xs.min())
    y1,x1 = int(ys.max())+1,int(xs.max())+1
    old = mem.get("clean")
    last = mem.get("last")
    if old is not None and last and last["kind"] == "move":
        patch = old[y:y1,x:x1].copy()
        patch[patch == 14] = 2
        a[y:y1,x:x1] = patch
    else:
        near = a[max(0,y-2):min(63,y1+2),
                 max(0,x-2):min(cut,x1+2)]
        colors = (2,13,8,9,15,12,1,6,7,10)
        color = max(colors, key=lambda c: (
            np.count_nonzero(near == c), c == 2))
        if not np.count_nonzero(near == color):
            color = 2
        a[y:y1,x:x1] = color
    return a


def buttons(obs, cut):
    result = []
    for color in range(16):
        if color in (0,3,4,5):
            continue
        mask = np.zeros((64,64), dtype=bool)
        mask[:63,cut+1:] = obs[:63,cut+1:] == color
        for y,x,y1,x1,pts in components(mask):
            h,w = y1-y,x1-x
            if len(pts) < 8 or h < 2 or w < 3:
                continue
            cy,cx = (y+y1-1)//2,(x+x1-1)//2
            if obs[cy,cx] != color:
                cy,cx = pts[len(pts)//2]
            result.append({
                "key":(color,y,x), "color":color,
                "y":y,"x":x,"h":h,"w":w,
                "button":(int(cy),int(cx)), "type":"toggle"
            })
    small = [b for b in result
             if b["h"] == 4 and b["w"] == 4
             and b["color"] in (9,10)]
    if len(small) >= 4:
        for color,labels in ((10,("left","right")),(9,("up","down"))):
            group = sorted([b for b in small if b["color"] == color],
                           key=lambda b:b["x"])
            if len(group) >= 2:
                group[0]["type"] = labels[0]
                group[-1]["type"] = labels[1]
        for b in result:
            if b["color"] == 8 and b["h"] <= 4 and b["w"] >= 7:
                b["type"] = "grab"
    return result


def teleports(a):
    tiles = {}
    for yy,xx in zip(*np.nonzero((a == 6)|(a == 7))):
        y,x = int(yy),int(xx)
        if y >= 62 or x >= 62:
            continue
        q = a[y:y+2,x:x+2]
        if (q[0,0] == q[1,1] and q[0,1] == q[1,0] and
                {int(q[0,0]),int(q[0,1])} == {6,7}):
            tiles[y*64+x] = int(q[0,0])
    out = {}
    for p,color in tiles.items():
        others = sorted(q for q,c in tiles.items() if q != p and c != color)
        if not others:
            others = sorted(q for q in tiles if q != p)
        if others:
            out[p] = others[0]
    return out


def compile_factor(f, cut):
    if "compiled" in f:
        return f["compiled"]
    maps = f["maps"]
    raw = [terrain(a,cut) for a in maps]
    region = np.zeros((64,64), dtype=bool)
    if len(maps) > 1:
        first_floor,first_block = raw[0]
        for i in range(1,len(maps)):
            region |= maps[i] != maps[0]
            # An unchanged opaque pixel can change collision mode.
            region |= raw[i][0] != first_floor
            region |= raw[i][1] != first_block
    region[:,cut:] = False
    region[63,:] = False
    f["region"] = region
    ss = [floor & region for floor,block in raw]
    bs = [block & region for floor,block in raw]
    ts = [teleports(a) for a in maps]
    f["compiled"] = (ss,bs,ts)
    return f["compiled"]


def signature(mem):
    m = mem.get("machine")
    return tuple(f["cur"] for f in mem["factors"]) + (
        m["cur"] if m else -1,)


def setup_machine(mem,a,controls):
    if mem.get("machine") or not controls:
        return
    cut = mem["cut"]
    cr = find_crane(a,cut)
    if cr:
        y,x,attached = cr
        objects = [c for c in components(a[:63,:cut] == 12)
                   if c[2]-c[0] == 12 and c[3]-c[1] == 6]
        if objects:
            py,px,_,_,pts = objects[0]
            shape = np.zeros((12,6), dtype=bool)
            for yy,xx in pts:
                shape[yy-py,xx-px] = True
            zeros = np.argwhere(a[:63,:cut] == 0)
            if len(zeros):
                ox = int(zeros[:,1].min())-1
                oy = int(zeros[:,0].max())-5
            else:
                oy,ox = py+14,px-18
            positions = [(0,0),(0,1),(0,2),(0,3),
                         (1,0),(2,0),(3,0),(1,3),(2,3),(3,3)]
            states = [(dx,up,flag) for flag in (False,True)
                      for dx,up in positions]
            state = (round((x-ox)/4),round((oy-y)/4),attached)
            cur = states.index(state) if state in states else 0
            fixed = (oy-14,ox+18) if attached else (py,px)
            mem["machine"] = {
                "kind":"standard", "cur":cur,
                "states":states,"origin":(oy,ox),
                "fixed":fixed,"object":shape,
                "controls":controls.copy(),"bad":set()
            }
            return
    mem["machine"] = {
        "kind":"learned","cur":0,"maps":[a.copy()],
        "edges":{},"controls":controls.copy(),"probes":0
    }


def ingest(mem,a,p):
    last = mem.get("last")
    if not last:
        return
    kind = last["kind"]
    if kind == "move":
        if p != last["target"]:
            mem["bans"].add((last["sig"],last["target"]))
            mem["plan"] = []
    elif kind == "factor":
        f = mem["factors"][last["i"]]
        if last.get("probe"):
            match = None
            for i,old in enumerate(f["maps"]):
                if np.array_equal(a,old):
                    match = i
                    break
            if match is None:
                f["maps"].append(a.copy())
                f["cur"] = len(f["maps"])-1
                if len(f["maps"]) >= 12:
                    f["complete"] = True
            else:
                f["cur"] = match
                f["complete"] = True
            f.pop("compiled",None)
            compile_factor(f,mem["cut"])
        else:
            compile_factor(f,mem["cut"])
            reg = f["region"]
            if reg.any():
                scores = [int(np.count_nonzero(a[reg] != im[reg]))
                          for im in f["maps"]]
                cur = min(range(len(scores)),key=lambda i:scores[i])
                f["cur"] = cur
                if cur != last["next"]:
                    mem["click_bans"].add(
                        (last["sig"],last["from"],last["i"]))
                    mem["plan"] = []
            else:
                f["cur"] = last["next"]
        if p != last["target"]:
            mem["plan"] = []
        if p != last["from"]:
            f.setdefault("observed_tp",{})[
                (last["old"],last["from"])] = p
    elif kind == "machine":
        m = mem["machine"]
        if m["kind"] == "standard":
            cr = find_crane(a,mem["cut"])
            expected = last["next"]
            if cr:
                y,x,attached = cr
                oy,ox = m["origin"]
                state = (round((x-ox)/4),round((oy-y)/4),attached)
                m["cur"] = (m["states"].index(state)
                            if state in m["states"] else expected)
            else:
                m["cur"] = expected
            if m["cur"] != expected:
                m["bad"].add((last["old"],last["control"]))
                mem["plan"] = []
        else:
            ignore = np.zeros((64,64), dtype=bool)
            for f in mem["factors"]:
                compile_factor(f,mem["cut"])
                ignore |= f["region"]
            keep = ~ignore
            match = None
            for i,old in enumerate(m["maps"]):
                if np.array_equal(a[keep],old[keep]):
                    match = i
                    break
            if match is None:
                match = len(m["maps"])
                m["maps"].append(a.copy())
            m["edges"][(last["old"],last["control"])] = match
            m["cur"] = match
            if last.get("probe"):
                m["probes"] += 1
            if match != last.get("next",match):
                mem["plan"] = []
        if p != last["target"]:
            mem["plan"] = []
    mem["last"] = None


def standard_models(m):
    ss,bs,edges = [],[],[]
    oy,ox = m["origin"]
    index = {s:i for i,s in enumerate(m["states"])}
    for i,(dx,up,attached) in enumerate(m["states"]):
        y,x = oy-4*up,ox+4*dx
        floor = np.zeros((64,64), dtype=bool)
        block = np.zeros((64,64), dtype=bool)
        shape = CR2 if attached else CR1
        if 0 <= y <= 55 and 0 <= x <= 55:
            block[y:y+8,x:x+8] = shape
        py,px = (y-2,x+6) if attached else m["fixed"]
        if 0 <= py <= 51 and 0 <= px <= 57:
            floor[py:py+12,px:px+6] |= m["object"]
        ss.append(floor)
        bs.append(block)
        es = []
        for control,ddx,dup in (
                ("up",0,1),("down",0,-1),
                ("left",-1,0),("right",1,0)):
            target = (dx+ddx,up+dup,attached)
            if (target in index and control in m["controls"] and
                    (i,control) not in m["bad"]):
                es.append((control,index[target]))
        fy,fx = m["fixed"]
        if (not attached and x+9 == fx+3 and y+4 == fy+6
                and "grab" in m["controls"]):
            es.append(("grab",index[(dx,up,True)]))
        edges.append(es)
    return ss,bs,edges


def world_models(mem,a):
    cut = mem["cut"]
    base_floor,base_block = terrain(a,cut)
    fs = []
    for f in mem["factors"]:
        compiled = compile_factor(f,cut)
        base_floor[f["region"]] = False
        base_block[f["region"]] = False
        fs.append(compiled)
    md = None
    m = mem.get("machine")
    if m:
        if m["kind"] == "standard":
            ss,bs,edges = standard_models(m)
            base_floor[ss[m["cur"]] | bs[m["cur"]]] = False
            base_block[bs[m["cur"]]] = False
            base_floor |= a == 2
            md = ss,bs,edges
        else:
            raw = [terrain(im,cut) for im in m["maps"]]
            region = np.zeros((64,64), dtype=bool)
            for i in range(1,len(raw)):
                region |= m["maps"][i] != m["maps"][0]
                region |= raw[i][0] != raw[0][0]
                region |= raw[i][1] != raw[0][1]
            for f in mem["factors"]:
                region &= ~f["region"]
            base_floor[region] = False
            base_block[region] = False
            ss = [fl & region for fl,bl in raw]
            bs = [bl & region for fl,bl in raw]
            edges = [
                [(c,d) for (s,c),d in m["edges"].items()
                 if s == i and d != i]
                for i in range(len(raw))
            ]
            md = ss,bs,edges
    return base_floor,base_block,fs,md


def reconstruct(prev,node,root):
    actions = []
    while node != root:
        old,action = prev[node]
        actions.append(action)
        node = old
    actions.reverse()
    return actions


def simple_path(mem,a,start,goal,refuge=False):
    fl,bl = terrain(a,mem["cut"])
    legal = legal_positions(fl,bl)
    sig = signature(mem)
    q = deque([start])
    prev = {start:None}
    target = None
    while q:
        p = q.popleft()
        if ((not refuge and p == goal) or
                (refuge and p != start and safe_position(a,p))):
            target = p
            break
        y,x = divmod(p,64)
        for name,dy,dx in DIRS:
            yy,xx = y+dy,x+dx
            if 0 <= yy < 63 and 0 <= xx < mem["cut"]-1:
                n = yy*64+xx
                if (legal[n] and n not in prev and
                        (sig,n) not in mem["bans"]):
                    prev[n] = (p,name)
                    q.append(n)
    if target is None:
        return None
    path = []
    p = target
    while prev[p] is not None:
        old,name = prev[p]
        path.append({
            "kind":"move","a":name,"target":p,"r":None,"c":None})
        p = old
    return path[::-1]


def search(mem,a,start,goal,visible):
    bf,bb,fs,md = world_models(mem,a)
    factors = mem["factors"]
    m = mem.get("machine")
    initial = signature(mem)
    cut = mem["cut"]
    union = bf.copy()
    reversed_tp = {}
    for ss,bs,ts in fs:
        for s in ss:
            union |= s
        for t in ts:
            for p,q in t.items():
                reversed_tp.setdefault(q,set()).add(p)
    if md:
        for s in md[0]:
            union |= s
    union[divmod(start,64)] = True
    if goal is not None:
        union[divmod(goal,64)] = True
    allowed = union.ravel()
    sy,sx = divmod(start,64)
    neighbors = {}
    for y in range(sy%2,63,2):
        for x in range(sx%2,cut-1,2):
            p = y*64+x
            if not allowed[p]:
                continue
            ns = []
            for name,dy,dx in DIRS:
                yy,xx = y+dy,x+dx
                if 0 <= yy < 63 and 0 <= xx < cut-1:
                    n = yy*64+xx
                    if allowed[n]:
                        ns.append((n,name))
            neighbors[p] = ns

    heuristic = {}
    if goal is not None:
        q = deque([goal])
        heuristic[goal] = 0
        while q:
            p = q.popleft()
            ns = [n for n,name in neighbors.get(p,())]
            ns += list(reversed_tp.get(p,()))
            for n in ns:
                if n not in heuristic:
                    heuristic[n] = heuristic[p]+1
                    q.append(n)

    def h(p):
        if p in heuristic:
            return heuristic[p]
        if goal is None:
            return 0
        y,x = divmod(p,64)
        gy,gx = divmod(goal,64)
        return (abs(y-gy)+abs(x-gx))/2

    cache = {}
    bans_by_state = {}
    for sig,p in mem["bans"]:
        bans_by_state.setdefault(sig,[]).append(p)

    def legal(conf):
        if conf not in cache:
            fl,bl = bf.copy(),bb.copy()
            for i,s in enumerate(conf[:-1]):
                fl |= fs[i][0][s]
                bl |= fs[i][1][s]
            if md:
                fl |= md[0][conf[-1]]
                bl |= md[1][conf[-1]]
            lm = legal_positions(fl,bl)
            for p in bans_by_state.get(conf,()):
                lm[p] = False
            cache[conf] = lm
        return cache[conf]

    pickups = set()
    for y,x,y1,x1,pts in components(a[:63,:cut] == 15):
        if y1-y <= 3 and x1-x <= 5 and len(pts) >= 4:
            for py in range(max(0,y-1),y1):
                for px in range(max(0,x-1),x1):
                    if py%2 == sy%2 and px%2 == sx%2:
                        pickups.add(py*64+px)

    root = initial,start
    dist = {root:0}
    prev = {}
    heap = [(h(start),0,0,root)]
    serial = 0
    expanded = 0
    answer = None
    best = None
    best_score = -1e30

    def push(conf,p,g,old,action):
        nonlocal serial
        node = conf,p
        if g < dist.get(node,10**9):
            dist[node] = g
            prev[node] = old,action
            serial += 1
            heapq.heappush(heap,(g+1.1*h(p),g,serial,node))

    while heap and expanded < 100000:
        _,g,_,node = heapq.heappop(heap)
        if dist.get(node) != g:
            continue
        expanded += 1
        conf,p = node
        if p == goal:
            answer = node
            break
        if p not in mem["visited"]:
            y,x = divmod(p,64)
            reward = 180 if p in pickups else 30
            if a[y,x] == 2:
                reward += 8
            score = reward - .8*g - .25*h(p)
            if score > best_score:
                best_score,best = score,node

        lm = legal(conf)
        for n,name in neighbors.get(p,()):
            if lm[n]:
                push(conf,n,g+1,node,{
                    "kind":"move","a":name,"target":n,"r":None,"c":None})

        for i,f in enumerate(factors):
            if not f["complete"] or len(f["maps"]) < 2:
                continue
            if f["gated"] and p not in f["enabled"]:
                continue
            if (conf,p,i) in mem["click_bans"]:
                continue
            j = conf[i]
            nj = (j+1)%len(f["maps"])
            nc = list(conf)
            nc[i] = nj
            nc = tuple(nc)
            dest = f.get("observed_tp",{}).get(
                (j,p),fs[i][2][j].get(p,p))
            if legal(nc)[dest]:
                r,c = f["button"]
                push(nc,dest,g+1,node,{
                    "kind":"factor","i":i,"old":j,"next":nj,
                    "a":"ACTION6","r":r,"c":c,"target":dest})

        if md:
            for control,nj in md[2][conf[-1]]:
                nc = conf[:-1]+(nj,)
                if legal(nc)[p]:
                    r,c = m["controls"][control]
                    push(nc,p,g+1,node,{
                        "kind":"machine","control":control,
                        "old":conf[-1],"next":nj,
                        "a":"ACTION6","r":r,"c":c,"target":p})

    if answer is not None:
        return reconstruct(prev,answer,root),None
    if best is not None:
        return None,reconstruct(prev,best,root)
    return None,None


def safe_position(a,p):
    y,x = divmod(p,64)
    if a[y,x] != 2:
        return False
    patch = a[max(0,y-1):min(63,y+3),max(0,x-1):min(64,x+3)]
    return np.count_nonzero(patch == 2) >= 5


def machine_experiment(mem,p):
    m = mem.get("machine")
    if (not m or m["kind"] != "learned" or
            m["probes"] >= 65 or len(m["maps"]) > 35):
        return None
    order = ("grab","up","right","down","left")
    current = m["cur"]
    q = deque([current])
    prev = {current:None}
    target = control = None
    while q:
        s = q.popleft()
        for c in order:
            if c in m["controls"] and (s,c) not in m["edges"]:
                target,control = s,c
                break
        if target is not None:
            break
        for (source,c),dest in m["edges"].items():
            if source == s and dest != s and dest not in prev:
                prev[dest] = s,c
                q.append(dest)
    if target is None:
        return None
    actions = []
    s = target
    while prev[s] is not None:
        old,c = prev[s]
        r,col = m["controls"][c]
        actions.append({
            "kind":"machine","control":c,"old":old,"next":s,
            "a":"ACTION6","r":r,"c":col,"target":p})
        s = old
    actions.reverse()
    r,col = m["controls"][control]
    actions.append({
        "kind":"machine","control":control,"old":target,"probe":True,
        "a":"ACTION6","r":r,"c":col,"target":p})
    return actions


def issue(mem,action,p,available):
    name = action["a"]
    if name not in available:
        mem["plan"] = []
        fallback = next((n for n,dy,dx in DIRS if n in available),
                        available[0])
        return fallback,None,None
    last = dict(action)
    last["from"] = p
    last["sig"] = signature(mem)
    if action["kind"] == "factor":
        f = mem["factors"][action["i"]]
        last["old"] = f["cur"]
    elif action["kind"] == "machine":
        last["old"] = mem["machine"]["cur"]
    mem["last"] = last
    return name,action.get("r"),action.get("c")


def follow(mem,path,p,available):
    mem["plan"] = path[1:]
    return issue(mem,path[0],p,available)


def act(obs, available, memory):
    obs = np.asarray(obs,dtype=np.int16)
    available = [str(a).split(".")[-1] for a in available]
    if "factors" not in memory:
        memory.update({
            "cut":get_cut(obs),"factors":[],"machine":None,
            "plan":[],"last":None,"bans":set(),"click_bans":set(),
            "visited":set(),"counts":{},"calls":0,"resets":0
        })
    memory["calls"] += 1
    p = get_player(obs)
    if p is None:
        return ("RESET",None,None) if "RESET" in available else (
            available[0],None,None)

    a = clean_observation(obs,memory,p)
    ingest(memory,a,p)
    memory["clean"] = a.copy()
    memory["visited"].add(p)
    node = signature(memory),p
    memory["counts"][node] = memory["counts"].get(node,0)+1

    goal = get_goal(obs,memory["cut"])
    if goal is not None:
        memory["goal"] = goal
    else:
        goal = memory.get("goal")

    panel = buttons(obs,memory["cut"])
    visible = {b["key"]:b for b in panel}
    controls = {b["type"]:b["button"]
                for b in panel if b["type"] != "toggle"}
    setup_machine(memory,a,controls)
    if memory.get("machine"):
        memory["machine"]["controls"].update(controls)

    existing = {f["key"]:i for i,f in enumerate(memory["factors"])}
    discovered = False
    for b in panel:
        if b["type"] != "toggle":
            continue
        if b["key"] not in existing:
            f = {
                "key":b["key"],"color":b["color"],"button":b["button"],
                "maps":[a.copy()],"cur":0,"complete":False,
                "enabled":{p},"gated":False,"started":False
            }
            memory["factors"].append(f)
            compile_factor(f,memory["cut"])
            discovered = True
        else:
            f = memory["factors"][existing[b["key"]]]
            f["button"] = b["button"]
            f["enabled"].add(p)
    for f in memory["factors"]:
        if f["key"] not in visible:
            f["gated"] = True
    if discovered:
        memory["plan"] = []

    if memory["plan"]:
        action = memory["plan"].pop(0)
        if action["kind"] == "factor":
            f = memory["factors"][action["i"]]
            if f["key"] in visible:
                return issue(memory,action,p,available)
            memory["plan"] = []
        else:
            return issue(memory,action,p,available)

    if goal is not None:
        path = simple_path(memory,a,p,goal)
        if path:
            return follow(memory,path,p,available)

    solution,exploration = search(memory,a,p,goal,visible)
    if solution:
        return follow(memory,solution,p,available)

    incomplete = [
        (i,f) for i,f in enumerate(memory["factors"])
        if not f["complete"] and f["key"] in visible
    ]
    if incomplete and "ACTION6" in available:
        if not safe_position(a,p):
            path = simple_path(memory,a,p,None,refuge=True)
            if path:
                return follow(memory,path,p,available)
        y,x = divmod(p,64)
        underneath = int(a[y,x])
        candidates = []
        for i,f in incomplete:
            danger = (
                f["color"] == underneath or
                (f["color"] == 8 and underneath == 13) or
                (f["color"] in (2,11) and underneath in (1,12)))
            candidates.append((danger,-len(f["maps"]),i,f))
        _,_,i,f = min(candidates,key=lambda t:t[:3])
        if not f["started"]:
            # Do not compare against a snapshot from before unrelated actions.
            f["maps"] = [a.copy()]
            f["cur"] = 0
            f["started"] = True
            f.pop("compiled",None)
            compile_factor(f,memory["cut"])
        r,c = f["button"]
        return issue(memory,{
            "kind":"factor","i":i,"probe":True,"old":f["cur"],
            "a":"ACTION6","r":r,"c":c,"target":p
        },p,available)

    if safe_position(a,p) and "ACTION6" in available:
        experiment = machine_experiment(memory,p)
        if experiment:
            return follow(memory,experiment,p,available)

    if exploration:
        return follow(memory,exploration,p,available)

    if not safe_position(a,p):
        path = simple_path(memory,a,p,None,refuge=True)
        if path:
            return follow(memory,path,p,available)

    # State-aware visit counts prevent oscillation when the model is incomplete.
    fl,bl = terrain(a,memory["cut"])
    legal = legal_positions(fl,bl)
    y,x = divmod(p,64)
    sig = signature(memory)
    choices = []
    for name,dy,dx in DIRS:
        yy,xx = y+dy,x+dx
        if name in available and 0 <= yy < 63 and 0 <= xx < memory["cut"]-1:
            n = yy*64+xx
            if legal[n] and (sig,n) not in memory["bans"]:
                score = memory["counts"].get((sig,n),0)*10
                if goal is not None:
                    gy,gx = divmod(goal,64)
                    score += .01*(abs(yy-gy)+abs(xx-gx))
                choices.append((score,name,n))
    if choices:
        _,name,n = min(choices)
        return issue(memory,{
            "kind":"move","a":name,"target":n,"r":None,"c":None
        },p,available)

    # Try a visible control once more if unmodelled gating stranded the player.
    if "ACTION6" in available:
        choices = []
        for i,f in enumerate(memory["factors"]):
            if f["key"] in visible:
                k = (sig,p,i)
                if k not in memory["click_bans"]:
                    choices.append((i,f))
        if choices:
            i,f = choices[0]
            memory["click_bans"].add((sig,p,i))
            r,c = f["button"]
            nj = (f["cur"]+1)%len(f["maps"])
            return issue(memory,{
                "kind":"factor","i":i,"old":f["cur"],"next":nj,
                "a":"ACTION6","r":r,"c":c,"target":p
            },p,available)

    if "RESET" in available:
        memory.clear()
        return "RESET",None,None
    return available[0],None,None
