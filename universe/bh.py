"""Barnes-Hut octree gravity in numba (O(N log N)). Stars (uniform softening) go in the tree;
the few galaxy cores (large halo softening) are summed directly, symmetric softening .5*(e_i^2+e_j^2)."""
import numpy as np, numba as nb

@nb.njit(cache=True)
def build(x, m, c, h, maxd=40):
    n = x.shape[0]; cap = 16*n+64
    child = -np.ones((cap, 8), np.int64); body = -np.ones(cap, np.int64); nxt = -np.ones(n, np.int64)
    cen = np.zeros((cap, 3)); half = np.zeros(cap); depth = np.zeros(cap, np.int64)
    cen[0] = c; half[0] = h; nn = 1
    for p in range(n):
        node = 0
        while True:
            if body[node] == -2:                       # internal: descend
                o = (x[p, 0] > cen[node, 0])*1 + (x[p, 1] > cen[node, 1])*2 + (x[p, 2] > cen[node, 2])*4
                if child[node, o] == -1:
                    k = nn; nn += 1; child[node, o] = k; hh = half[node]*.5; half[k] = hh; depth[k] = depth[node]+1
                    cen[k, 0] = cen[node, 0] + (hh if o & 1 else -hh); cen[k, 1] = cen[node, 1] + (hh if o & 2 else -hh)
                    cen[k, 2] = cen[node, 2] + (hh if o & 4 else -hh); body[k] = p; break
                node = child[node, o]
            elif body[node] == -1:
                body[node] = p; break
            elif depth[node] >= maxd:                  # bucket (coincident points): linked list
                nxt[p] = body[node]; body[node] = p; break
            else:                                      # split leaf
                q = body[node]; body[node] = -2
                o = (x[q, 0] > cen[node, 0])*1 + (x[q, 1] > cen[node, 1])*2 + (x[q, 2] > cen[node, 2])*4
                k = nn; nn += 1; child[node, o] = k; hh = half[node]*.5; half[k] = hh; depth[k] = depth[node]+1
                cen[k, 0] = cen[node, 0] + (hh if o & 1 else -hh); cen[k, 1] = cen[node, 1] + (hh if o & 2 else -hh)
                cen[k, 2] = cen[node, 2] + (hh if o & 4 else -hh); body[k] = q; nxt[q] = -1
    mass = np.zeros(nn); com = np.zeros((nn, 3))
    for node in range(nn-1, -1, -1):                   # children always have larger index -> reverse = post-order
        if body[node] == -2:
            for o in range(8):
                k = child[node, o]
                if k >= 0: mass[node] += mass[k]; com[node] += mass[k]*com[k]
        else:
            p = body[node]
            while p >= 0: mass[node] += m[p]; com[node] += m[p]*x[p]; p = nxt[p]
        if mass[node] > 0: com[node] /= mass[node]
    return child[:nn], body[:nn], nxt, half[:nn], mass, com

@nb.njit(parallel=True, fastmath=True, cache=True)
def walk(t, te2, x, m, child, body, nxt, half, mass, com, se2, theta):
    nt = t.shape[0]; a = np.zeros((nt, 3)); th2 = theta*theta
    for i in nb.prange(nt):
        stack = np.empty(1024, np.int64); stack[0] = 0; sp = 1; ax = ay = az = 0.; e2 = .5*(te2[i]+se2)
        while sp > 0:
            sp -= 1; node = stack[sp]
            if mass[node] == 0: continue
            if body[node] == -2:
                dx = com[node, 0]-t[i, 0]; dy = com[node, 1]-t[i, 1]; dz = com[node, 2]-t[i, 2]; d2 = dx*dx+dy*dy+dz*dz
                if 4*half[node]*half[node] > th2*d2:
                    for o in range(8):
                        k = child[node, o]
                        if k >= 0: stack[sp] = k; sp += 1
                else:
                    r2 = d2+e2; f = mass[node]/(r2*np.sqrt(r2)); ax += f*dx; ay += f*dy; az += f*dz
            else:
                p = body[node]
                while p >= 0:
                    dx = x[p, 0]-t[i, 0]; dy = x[p, 1]-t[i, 1]; dz = x[p, 2]-t[i, 2]; d2 = dx*dx+dy*dy+dz*dz
                    if d2 > 0.:
                        r2 = d2+e2; f = m[p]/(r2*np.sqrt(r2)); ax += f*dx; ay += f*dy; az += f*dz
                    p = nxt[p]
        a[i, 0] = ax; a[i, 1] = ay; a[i, 2] = az
    return a

@nb.njit(parallel=True, fastmath=True, cache=True)
def direct_src(t, te2, xs, ms, se2):
    a = np.zeros((t.shape[0], 3))
    for i in nb.prange(t.shape[0]):
        for j in range(xs.shape[0]):
            dx = xs[j, 0]-t[i, 0]; dy = xs[j, 1]-t[i, 1]; dz = xs[j, 2]-t[i, 2]; d2 = dx*dx+dy*dy+dz*dz
            if d2 > 0.:
                r2 = d2+.5*(te2[i]+se2[j]); f = ms[j]/(r2*np.sqrt(r2)); a[i, 0] += f*dx; a[i, 1] += f*dy; a[i, 2] += f*dz
    return a

def accel_bh(x, m, e2, cores, stars, star_e2, theta=0.6):
    """Total acceleration: tree over stars + direct sum over cores."""
    xs = np.ascontiguousarray(x[stars]); lo, hi = xs.min(0), xs.max(0); T = build(xs, m[stars], .5*(lo+hi), .5*(hi-lo).max()*1.001+1e-9)
    a = walk(x, e2, xs, m[stars], *T[:2], T[2], *T[3:], star_e2, theta)
    a += direct_src(x, e2, np.ascontiguousarray(x[cores]), m[cores], e2[cores])
    return a
