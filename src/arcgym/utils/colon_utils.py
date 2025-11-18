#utilities for extracting/computing colon mesh geometry
#only expect one instance of geom, batched geom is not considered here.
import torch


def extract_bbox(nodal_pos):
    '''
    The indices of output coordinates for bounding box corners are illustrated as
            1-------2
           /|      /|
          / |     / |
         6--|----7  |
         |  |    |  |
         |  0----|--3
         | /     | /
         |/      |/
         5-------4   
    '''

    min_coords = torch.min(nodal_pos, dim=0).values
    max_coords = torch.max(nodal_pos, dim=0).values
    
    return torch.tensor([[min_coords[0], min_coords[1], min_coords[2]],
                         [min_coords[0], min_coords[1], max_coords[2]],
                         [min_coords[0], max_coords[1], max_coords[2]],
                         [min_coords[0], max_coords[1], min_coords[2]],
                         [max_coords[0], max_coords[1], min_coords[2]],
                         [max_coords[0], min_coords[1], min_coords[2]],
                         [max_coords[0], min_coords[1], max_coords[2]],
                         [max_coords[0], max_coords[1], max_coords[2]],
                        ]).to(device=nodal_pos.device)

def extract_attach_nodals(nodal_pos):
    #return the indices for rectum, descend, splenic, heptic, cecum to create kinematic constraints
    #assuming z is the vertical direction, the colon is largely staying in the x-y plane
    corners = extract_bbox(nodal_pos)

    #rectum, lowest and around the centerline
    dist = torch.linalg.norm(nodal_pos - (corners[0]+corners[3]+corners[4]+corners[5])/4., dim=-1)
    rectum_idx = torch.argmin(dist)
    
    #descend, close to middle on the right
    dist = torch.linalg.norm(nodal_pos - (corners[2]+corners[3]+corners[4]+corners[7])/4., dim=-1)
    descend_idx = torch.argmin(dist)

    #splenic, close to the right highest point
    dist = torch.linalg.norm(nodal_pos - (corners[2]+corners[7])/2., dim=-1)
    splenic_idx = torch.argmin(dist)

    #heptic, close to the left highest point
    dist = torch.linalg.norm(nodal_pos - (corners[1]+corners[6])/2., dim=-1)
    heptic_idx = torch.argmin(dist)

    #cecum, this is probably annoying, middle close left?
    tmp1 = (corners[0]+corners[1]+corners[5]+corners[6])/4.
    tmp2 = (corners[2]+corners[3]+corners[4]+corners[7])/4.

    dist = torch.linalg.norm(nodal_pos - tmp1, dim=-1)
    cecum_idx = torch.argmin(dist)

    return rectum_idx, descend_idx, splenic_idx, heptic_idx, cecum_idx