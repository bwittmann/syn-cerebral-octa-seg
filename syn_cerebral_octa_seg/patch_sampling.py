"""File containing functions used to extract patches from corrosion casts."""

import numpy as np
from tqdm import tqdm


def angle_between(v1, v2):
    def unit_vector(vectors):
        return vectors / np.linalg.norm(vectors, axis=1)[None].T
    
    v1_u, v2_u = unit_vector(v1), unit_vector(v2)
    dot_prod = (v1_u* v2_u).sum(1)
    angles = np.rad2deg(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
    angles = np.absolute((angles - 90))                     # horizental vessels should be 0deg
    angles_intensities = (np.absolute((angles - 90))) / 90  # 1 horizontal, 0 vertical
    return angles_intensities

def divide_graph(args, path_to_graph, volume_size):
    if args.cc: # load corrosion casts
        edge_index = np.load(path_to_graph / 'edges.npy').astype('int64')
        edge_rad = np.load(path_to_graph / 'radius_for_edges.npy')
        pos = np.load(path_to_graph / 'vertices.npy')
        nodes_region = np.load(path_to_graph / 'region_for_vertices.npy').astype('int64')
    else:
        raise NotImplementedError

    # divide graph into patches
    x_max, y_max, z_max = pos.max(axis=0)    # x, y, z
    z_max = 3000 # hard threshold of z-max, since solely interested in cortex
    x_range, y_range = np.arange(0, x_max, volume_size[0]), np.arange(0, y_max, volume_size[1])
    xg, yg = np.meshgrid(x_range, y_range, indexing='ij')

    patches = []
    for x, y in tqdm(zip(xg.reshape(-1), yg.reshape(-1)), 'Extract graph patches.', disable=args.dt):
        # construct pillar
        pillar_start = np.array((x, y, 0))
        pillar_end = np.array((x + volume_size[0], y + volume_size[1], z_max))

        if (pillar_start > np.array((x_max, y_max, z_max))).any(): # pillar start point should lie in the volume
            continue
        
        # find nodes in pillar
        node_ids = np.concatenate(((pos > pillar_start).all(axis=1)[None], (pos < pillar_end).all(axis=1)[None])).all(axis=0).nonzero()[0]
        node_ids = node_ids[nodes_region[node_ids].nonzero()]   # get rid of artifact node ids

        if len(node_ids) < 100: # skip patches with no/little nodes inside
            continue

        # extract position of surfacing vessels
        involved_vessel_mask = np.isin(edge_index, node_ids).all(axis=-1)
        involved_vessels = edge_index[involved_vessel_mask]
        involved_vessels_rad = edge_rad[involved_vessel_mask]
        involved_surfacing_vessels = involved_vessels[involved_vessels_rad > 13]

        if len(involved_surfacing_vessels) == 0:    # just utilize scans with surfacing vessels (helps to exclude artifacts)
            continue

        # determine position of top surfacing vessels to estimate FOV
        top_vessel_pos = pos[involved_surfacing_vessels.flatten()][:, -1].min()

        # crop pillar to patch with similar FOV than OCT brain scans
        patch_start = np.array((x, y, top_vessel_pos - involved_vessels_rad.max()))
        patch_end =  np.array((x + volume_size[0], y + volume_size[1], patch_start[2] + volume_size[2]))

        if patch_end[-1] > 3000:    # we are solely intereted in cortex
            continue

        # find nodes in patch
        node_ids = np.concatenate(((pos > patch_start).all(axis=1)[None], (pos < patch_end).all(axis=1)[None])).all(axis=0).nonzero()[0]

        # translate nodes and extract edges + rad
        edges_mask = np.isin(edge_index, node_ids).any(axis=1)
        edges, rad = edge_index[edges_mask], edge_rad[edges_mask]

        node_ids = np.unique(edges)
        nodes = (pos[node_ids] - patch_start) # patch_start == origin

        if len(edges) < 2000:
            continue

        new_node_ids = dict(np.concatenate((node_ids[None], np.arange(0, len(node_ids))[None])).T)
        edges = np.vectorize(new_node_ids.get)(edges)

        edge_vectors = nodes[edges[:, 1]] - nodes[edges[:, 0]]
        angles = angle_between(edge_vectors + 1e-8, np.repeat(np.array([[0, 0, 1]]), edges.shape[0], axis=0))

        patches.append((nodes, edges, rad, angles))

    print(f'Extracted {len(patches)} patches.')
    return patches