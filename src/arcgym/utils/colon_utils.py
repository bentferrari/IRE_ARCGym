#utilities for extracting/computing colon mesh geometry
#only expect one instance of geom, batched geom is not considered here.
import torch
import os
import ast


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


def load_attachment_indices_from_files(base_path=None, model_id="0000"):
    """
    Load attachment vertex indices from txt files.

    Args:
        base_path: Path to directory containing txt files. If None, uses default colon assets path.
        model_id: Model identifier (default "0000")

    Returns:
        Dictionary containing lists of vertex indices for each attachment region:
        {
            'rectum': [idx1, idx2, ...],
            'descending': [idx1, idx2, ...],
            'splenic': [idx1, idx2, ...],
            'hepatic': [idx1, idx2, ...],
            'ascending': [idx1, idx2, ...]
        }
    """
    if base_path is None:
        # Get the default path to the colon assets directory
        current_file = os.path.abspath(__file__)
        utils_dir = os.path.dirname(current_file)
        arcgym_dir = os.path.dirname(utils_dir)
        base_path = os.path.join(arcgym_dir, "assets", "colons")

    attachment_regions = {
        'rectum': f"{model_id}_rectum.txt",
        'descending': f"{model_id}_descending.txt",
        'splenic': f"{model_id}_splenic.txt",
        'hepatic': f"{model_id}_hepatic.txt",
        'ascending': f"{model_id}_ascending.txt"
    }

    attachment_indices = {}

    for region, filename in attachment_regions.items():
        filepath = os.path.join(base_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Attachment indices file not found: {filepath}")

        with open(filepath, 'r') as f:
            content = f.read().strip()
            # Parse the Python list format using ast.literal_eval
            indices = ast.literal_eval(content)
            attachment_indices[region] = indices

    return attachment_indices


def get_attachment_nodal_indices(base_path=None, model_id="0000", device='cpu'):
    """
    Load attachment vertex indices from txt files and return as a single tensor.

    Args:
        base_path: Path to directory containing txt files. If None, uses default colon assets path.
        model_id: Model identifier (default "0000")
        device: torch device to place the tensor on

    Returns:
        torch.Tensor: 1D tensor containing all attachment vertex indices (combined from all regions)
    """
    attachment_indices = load_attachment_indices_from_files(base_path, model_id)

    # Combine all indices from all regions
    all_indices = []
    for region in ['rectum', 'descending', 'splenic', 'hepatic', 'ascending']:
        all_indices.extend(attachment_indices[region])

    # Remove duplicates and convert to tensor
    unique_indices = sorted(list(set(all_indices)))

    return torch.tensor(unique_indices, dtype=torch.long, device=device)


def map_visual_to_simulation_nodes(visual_mesh_path, visual_vertex_indices, simulation_nodal_positions):
    """
    Map visual mesh vertex indices to simulation mesh node indices.

    The txt files contain indices for the high-resolution visual mesh, but PhysX
    uses a lower-resolution simulation mesh. This function finds the closest
    simulation node for each visual vertex.

    Args:
        visual_mesh_path: Path to the OBJ file containing the visual mesh
        visual_vertex_indices: List or tensor of vertex indices in the visual mesh
        simulation_nodal_positions: Tensor of shape (num_sim_nodes, 3) containing simulation node positions

    Returns:
        torch.Tensor: Indices of simulation nodes that correspond to the visual vertices
    """
    import trimesh

    # Load the visual mesh to get vertex positions
    try:
        visual_mesh = trimesh.load(visual_mesh_path)
        visual_vertices = torch.tensor(visual_mesh.vertices, dtype=torch.float32, device=simulation_nodal_positions.device)
    except:
        # Fallback: parse OBJ file manually
        visual_vertices = []
        with open(visual_mesh_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    visual_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        visual_vertices = torch.tensor(visual_vertices, dtype=torch.float32, device=simulation_nodal_positions.device)

    # Convert visual indices to tensor
    if not isinstance(visual_vertex_indices, torch.Tensor):
        visual_vertex_indices = torch.tensor(visual_vertex_indices, dtype=torch.long, device=simulation_nodal_positions.device)

    # Filter out invalid indices (indices >= num_vertices in visual mesh)
    num_visual_vertices = visual_vertices.shape[0]
    valid_mask = visual_vertex_indices < num_visual_vertices
    valid_indices = visual_vertex_indices[valid_mask]

    num_invalid = (~valid_mask).sum().item()
    if num_invalid > 0:
        print(f"Warning: {num_invalid} vertex indices out of bounds (max valid index: {num_visual_vertices-1})")
        print(f"Filtered to {len(valid_indices)} valid indices")

    # Get positions of the visual vertices we want to attach
    target_visual_positions = visual_vertices[valid_indices]  # Shape: (num_valid_visual_vertices, 3)

    # The visual mesh might be in a different scale than the simulation mesh
    # Apply scale factor of 0.001 (from COLON_GEOM_MESH_CFG) or 0.01 (from COLON_GEOM_MESH_CFG_endoscope)
    # Check which scale makes sense by comparing bounds
    visual_bbox_size = (target_visual_positions.max(dim=0).values - target_visual_positions.min(dim=0).values).max()
    sim_bbox_size = (simulation_nodal_positions.max(dim=0).values - simulation_nodal_positions.min(dim=0).values).max()

    # Estimate scale factor
    if visual_bbox_size > 0:
        estimated_scale = (sim_bbox_size / visual_bbox_size).item()
        print(f"Visual mesh bbox size: {visual_bbox_size:.3f}, Sim mesh bbox size: {sim_bbox_size:.3f}")
        print(f"Estimated scale factor: {estimated_scale:.6f}")

        # Apply scale to visual positions
        target_visual_positions = target_visual_positions * estimated_scale

    # Find closest simulation node for each visual vertex
    # Use broadcasting to compute all pairwise distances
    # target_visual_positions: (num_visual, 3)
    # simulation_nodal_positions: (num_sim_nodes, 3)

    # Move to CPU for large distance computation to avoid CUDA memory issues
    target_visual_cpu = target_visual_positions.cpu()
    sim_nodal_cpu = simulation_nodal_positions.cpu()

    distances = torch.cdist(target_visual_cpu, sim_nodal_cpu)  # Shape: (num_visual, num_sim_nodes)
    closest_sim_nodes = torch.argmin(distances, dim=1)  # Shape: (num_visual,)

    # Move back to original device
    closest_sim_nodes = closest_sim_nodes.to(simulation_nodal_positions.device)

    # Remove duplicates - multiple visual vertices may map to the same simulation node
    unique_sim_nodes = torch.unique(closest_sim_nodes)

    return unique_sim_nodes