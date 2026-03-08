#!/usr/bin/env python3
"""
Collaborative LEGO Structure Planner

A system for planning LEGO block placements based on spatial constraints
from multiple viewpoints. Supports different block types, viewing directions,
and both strict and relaxed adjacency modes.
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from itertools import product
from dataclasses import dataclass
from enum import Enum


class BlockType(Enum):
    SQUARE = "Square"  
    RECTANGLE = "Rectangle"
    ONE_CURVE = "OneCurve"
    TWO_CURVE = "TwoCurve"


class Color(Enum):
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    YELLOW = "Yellow"
    

class Direction(Enum):
    LEFT = "Left"
    RIGHT = "Right"
    ABOVE = "Above"
    BELOW = "Below"
    FRONT = "Front"
    BACK = "Back"


class ViewSide(Enum):
    NORTH = "North" 
    SOUTH = "South" 
    EAST = "East" 
    WEST = "West" 


class Orientation(Enum):
    HORIZONTAL = "Horizontal"
    VERTICAL = "Vertical"   


@dataclass
class Block:
    id: str
    color: Color
    block_type: BlockType
    orientation: Optional[Orientation] = None
    
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Returns (width, depth, height) of the block"""
        if self.block_type == BlockType.SQUARE:
            return (2, 2, 1)
        elif self.block_type == BlockType.RECTANGLE or self.block_type == BlockType.TWO_CURVE:
            if self.orientation == Orientation.HORIZONTAL:
                return (4, 2, 1)
            elif self.orientation == Orientation.VERTICAL:
                return (2, 4, 1)
            else:
                # Default to horizontal if no orientation set
                return (4, 2, 1)
        elif self.block_type == BlockType.ONE_CURVE:
            if self.orientation == Orientation.HORIZONTAL:
                return (3, 2, 1)
            elif self.orientation == Orientation.VERTICAL:
                return (2, 3, 1)
            else:
                # Default to horizontal if no orientation set
                return (3, 2, 1)
        return (0, 0, 0)
    
    def get_possible_orientations(self) -> List[Orientation]:
        """Get list of possible orientations for this block"""
        if self.block_type == BlockType.SQUARE:
            # Squares are symmetric, orientation doesn't matter
            return [Orientation.HORIZONTAL]
        elif self.orientation is not None:
            # Orientation is already specified
            return [self.orientation]
        else:
            # Try both orientations for rectangular blocks
            return [Orientation.HORIZONTAL, Orientation.VERTICAL]
    
    def with_orientation(self, orientation: Orientation) -> 'Block':
        """Create a copy of this block with specified orientation"""
        return Block(self.id, self.color, self.block_type, orientation)
    
    def get_occupied_positions(self, base_pos: 'Position') -> Set['Position']:
        """Get all unit positions occupied by this block"""
        positions = set()
        width, depth, height = self.get_dimensions()
        
        for x in range(width):
            for y in range(depth):
                for z in range(height):
                    positions.add(Position(base_pos.x + x, base_pos.y + y, base_pos.z + z))
        
        return positions
    
    def get_num_positions(self) -> int:
        """Get the number of unit positions this block occupies"""
        w, d, h = self.get_dimensions()
        return w * d * h


@dataclass
class Position:
    x: int
    y: int
    z: int
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z


@dataclass
class Constraint:
    """Represents a spatial relationship between blocks from a director's view"""
    block1_id: str
    relation: Direction
    block2_id: str
    director_id: int


@dataclass
class AbsoluteConstraint:
    """Represents an absolute position constraint for a block"""
    block_id: str
    position: Position
    director_id: int


@dataclass
class DirectorConfig:
    """Configuration for a director's viewpoint"""
    director_id: int
    view_side: ViewSide


@dataclass
class Placement:
    """Result of the solve() method containing placement and visualization"""
    placements: Dict[str, Position]
    field: List[List[List[str]]]  # 3D array [z][y][x] of block characters


class LegoPlanner:
    def __init__(self, grid_width: int = 30, grid_depth: int = 30, grid_height: int = 10):
        self.blocks: Dict[str, Block] = {}
        self.constraints: List[Constraint] = []
        self.absolute_constraints: List[AbsoluteConstraint] = []
        self.director_configs: Dict[int, DirectorConfig] = {}
        self.placed_blocks: Dict[str, Position] = {}  
        self.occupied_positions: Set[Position] = set()  
        self.strict_adjacent: bool = False
        # Grid boundaries
        self.grid_width = grid_width
        self.grid_depth = grid_depth  
        self.grid_height = grid_height

    def add_block(self, block: Block):
        """Add a block to the structure"""
        self.blocks[block.id] = block
    
    def add_director_config(self, director_id: int, view_side: ViewSide):
        """Configure which side a director is viewing from"""
        self.director_configs[director_id] = DirectorConfig(director_id, view_side)
        
    def add_constraint(self, constraint: Constraint):
        """Add a spatial constraint between blocks"""
        self.constraints.append(constraint)
    
    def add_absolute_constraint(self, constraint: AbsoluteConstraint):
        """Add an absolute position constraint for a block"""
        self.absolute_constraints.append(constraint)
        
    def parse_proposition(self, prop: str) -> Optional[Union[Constraint, AbsoluteConstraint]]:
        """Parse a proposition into a constraint
        Examples:
        - 'D1 Green_1 Left Red_1' -> relative constraint
        - 'D1 Green_1 At 2,3,0' -> absolute position constraint
        """
        parts = prop.split()
        if len(parts) < 3:
            return None
        
        director = int(parts[0][1:])  
        block1 = parts[1]
        
        # Check for absolute position constraint
        if len(parts) == 4 and parts[2].upper() == "AT":
            try:
                coords = parts[3].split(',')
                if len(coords) == 3:
                    x, y, z = int(coords[0]), int(coords[1]), int(coords[2])
                    return AbsoluteConstraint(block1, Position(x, y, z), director)
            except ValueError:
                return None
        
        # Otherwise, parse as relative constraint
        if len(parts) == 4:
            relation = Direction[parts[2].upper()]
            block2 = parts[3]
            return Constraint(block1, relation, block2, director)
        
        return None
    
    def _transform_direction_to_world(self, direction: Direction, view_side: ViewSide) -> Direction:
        """Transform a direction from a director's view to world coordinates"""
        # Define transformation mappings for each viewing direction
        transformations = {
            ViewSide.NORTH: {  
                Direction.LEFT: Direction.LEFT,
                Direction.RIGHT: Direction.RIGHT,
                Direction.ABOVE: Direction.ABOVE,
                Direction.BELOW: Direction.BELOW,
                Direction.FRONT: Direction.BACK, 
                Direction.BACK: Direction.FRONT   
            },
            ViewSide.SOUTH: {  
                Direction.LEFT: Direction.RIGHT,   
                Direction.RIGHT: Direction.LEFT,
                Direction.ABOVE: Direction.ABOVE,
                Direction.BELOW: Direction.BELOW,
                Direction.FRONT: Direction.FRONT, 
                Direction.BACK: Direction.BACK     
            },
            ViewSide.EAST: {  
                Direction.LEFT: Direction.FRONT,   
                Direction.RIGHT: Direction.BACK,  
                Direction.ABOVE: Direction.ABOVE,
                Direction.BELOW: Direction.BELOW,
                Direction.FRONT: Direction.RIGHT,  
                Direction.BACK: Direction.LEFT    
            },
            ViewSide.WEST: { 
                Direction.LEFT: Direction.BACK,    
                Direction.RIGHT: Direction.FRONT,  
                Direction.ABOVE: Direction.ABOVE,
                Direction.BELOW: Direction.BELOW,
                Direction.FRONT: Direction.LEFT,  
                Direction.BACK: Direction.RIGHT   
            }
        }
        
        return transformations[view_side][direction]
    
    def check_consistency(self) -> Tuple[bool, List[str]]:
        """Check if all constraints can be satisfied"""
        errors = []
        
        # Check if all referenced blocks exist
        for constraint in self.constraints:
            if constraint.block1_id not in self.blocks:
                errors.append(f"Block {constraint.block1_id} not defined")
            if constraint.block2_id not in self.blocks:
                errors.append(f"Block {constraint.block2_id} not defined")
        
        for abs_constraint in self.absolute_constraints:
            if abs_constraint.block_id not in self.blocks:
                errors.append(f"Block {abs_constraint.block_id} not defined")
        
        # Check if all directors have view configurations
        for constraint in self.constraints:
            if constraint.director_id not in self.director_configs:
                errors.append(f"Director {constraint.director_id} view side not configured")
        
        # Check for conflicting absolute constraints
        abs_map = {}
        for ac in self.absolute_constraints:
            if ac.block_id in abs_map:
                if abs_map[ac.block_id].position != ac.position:
                    errors.append(f"Conflicting absolute positions for {ac.block_id}")
            abs_map[ac.block_id] = ac
        
        return len(errors) == 0, errors
    
    def get_block_boundaries(self, block_id: str, base_pos: Position) -> Tuple[int, int, int, int, int, int]:
        """Get the bounding box of a block (min_x, max_x, min_y, max_y, min_z, max_z)"""
        block = self.blocks[block_id]
        w, d, h = block.get_dimensions()
        return (base_pos.x, base_pos.x + w - 1, 
                base_pos.y, base_pos.y + d - 1, 
                base_pos.z, base_pos.z + h - 1)
    
    def blocks_are_adjacent(self, block1_id: str, pos1: Position, 
                           block2_id: str, pos2: Position) -> bool:
        """Check if two blocks are adjacent (sharing a face)"""
        positions1 = self.blocks[block1_id].get_occupied_positions(pos1)
        positions2 = self.blocks[block2_id].get_occupied_positions(pos2)
        
        # Check if any positions are adjacent
        for p1 in positions1:
            for p2 in positions2:
                # Adjacent means sharing a face (differ by 1 in exactly one dimension)
                dx = abs(p1.x - p2.x)
                dy = abs(p1.y - p2.y)
                dz = abs(p1.z - p2.z)

                if self.strict_adjacent:
                    # Strict adjacency: must differ by 1 in exactly one dimension
                    if (dx == 1 and dy == 0 and dz == 0) or \
                       (dx == 0 and dy == 1 and dz == 0) or \
                       (dx == 0 and dy == 0 and dz == 1):
                        return True
                else:
                    # Can be more than one block apart in one dimension
                    if (dx >= 1 and dy == 0 and dz == 0) or \
                       (dx == 0 and dy >= 1 and dz == 0) or \
                       (dx == 0 and dy == 0 and dz >= 1):
                        return True
        
        return False
    
    def get_relative_direction(self, block1_id: str, pos1: Position,
                            block2_id: str, pos2: Position) -> Optional[Direction]:
        """Determine the direction of block1 relative to block2 in world coordinates"""
        bounds1 = self.get_block_boundaries(block1_id, pos1)
        bounds2 = self.get_block_boundaries(block2_id, pos2)
        
        # Extract bounds for readability
        # bounds format: (min_x, max_x, min_y, max_y, min_z, max_z)
        b1_min_x, b1_max_x, b1_min_y, b1_max_y, b1_min_z, b1_max_z = bounds1
        b2_min_x, b2_max_x, b2_min_y, b2_max_y, b2_min_z, b2_max_z = bounds2
        
        # Check for LEFT/RIGHT relationships (X-axis)
        # Blocks must align in Y dimension (overlap in Y coordinates)
        y_overlap = not (b1_max_y < b2_min_y or b2_max_y < b1_min_y)
        z_overlap = not (b1_max_z < b2_min_z or b2_max_z < b1_min_z)
        
        if y_overlap and z_overlap:
            if self.strict_adjacent:
                # Strict: blocks must be exactly adjacent (touching)
                if b2_max_x + 1 == b1_min_x:
                    return Direction.RIGHT  # block1 is RIGHT of block2
                elif b1_max_x + 1 == b2_min_x:
                    return Direction.LEFT   # block1 is LEFT of block2
            else:
                # Relaxed: blocks just need to be separated in X dimension
                if b1_min_x > b2_max_x:
                    return Direction.RIGHT  # block1 is RIGHT of block2
                elif b1_max_x < b2_min_x:
                    return Direction.LEFT   # block1 is LEFT of block2
        
        # Check for FRONT/BACK relationships (Y-axis)
        # Blocks must align in X dimension (overlap in X coordinates)
        x_overlap = not (b1_max_x < b2_min_x or b2_max_x < b1_min_x)
        
        if x_overlap and z_overlap:
            if self.strict_adjacent:
                # Strict: blocks must be exactly adjacent (touching)
                if b2_max_y + 1 == b1_min_y:
                    return Direction.BACK   # block1 is BACK of block2
                elif b1_max_y + 1 == b2_min_y:
                    return Direction.FRONT  # block1 is FRONT of block2
            else:
                # Relaxed: blocks just need to be separated in Y dimension
                if b1_min_y > b2_max_y:
                    return Direction.BACK   # block1 is BACK of block2
                elif b1_max_y < b2_min_y:
                    return Direction.FRONT  # block1 is FRONT of block2
        
        # Check for ABOVE/BELOW relationships (Z-axis)
        # Blocks must overlap in both X and Y dimensions
        if x_overlap and y_overlap:
            if self.strict_adjacent:
                # Strict: blocks must be exactly adjacent (touching)
                if b2_max_z + 1 == b1_min_z:
                    return Direction.ABOVE  # block1 is ABOVE block2
                elif b1_max_z + 1 == b2_min_z:
                    return Direction.BELOW  # block1 is BELOW block2
            else:
                # Relaxed: blocks just need to be separated in Z dimension
                if b1_min_z > b2_max_z:
                    return Direction.ABOVE  # block1 is ABOVE block2
                elif b1_max_z < b2_min_z:
                    return Direction.BELOW  # block1 is BELOW block2
            
        return None
    
    def solve(self) -> Tuple[bool, Optional[Placement]]:
        """
        Attempt to find a valid placement of all blocks that satisfies constraints
        Returns (success, Placement object) where Placement contains placements dict and 3D field
        """
        # Check basic consistency first
        consistent, errors = self.check_consistency()
        if not consistent:
            print("Consistency errors:", errors)
            return False, None
        
        # Transform all constraints to world coordinates
        world_constraints = []
        if self.director_configs:  # Only transform if we have director configs
            print("\nTransforming constraints to world coordinates:")
            for constraint in self.constraints:
                if constraint.director_id in self.director_configs:
                    view_side = self.director_configs[constraint.director_id].view_side
                    world_direction = self._transform_direction_to_world(constraint.relation, view_side)
                    world_constraint = Constraint(
                        constraint.block1_id,
                        world_direction,
                        constraint.block2_id,
                        constraint.director_id
                    )
                    print(f"  D{constraint.director_id} ({view_side.value}): {constraint.block1_id} {constraint.relation.value} {constraint.block2_id}")
                    print(f"    → World: {constraint.block1_id} {world_direction.value} {constraint.block2_id}")
                    world_constraints.append(world_constraint)
                else:
                    # Default to north view if not configured
                    print(f"  D{constraint.director_id} (default North): {constraint.block1_id} {constraint.relation.value} {constraint.block2_id}")
                    world_constraints.append(constraint)
            
            # Replace constraints with world-coordinate versions
            original_constraints = self.constraints
            self.constraints = world_constraints
        else:
            # No director configs, use constraints as-is
            original_constraints = self.constraints
        
        # Generate all possible orientation combinations
        orientation_options = []
        block_ids = list(self.blocks.keys())
        
        for block_id in block_ids:
            block = self.blocks[block_id]
            orientation_options.append(block.get_possible_orientations())
        
        # Try all combinations of orientations
        for orientations in product(*orientation_options):
            # Apply orientations to blocks
            oriented_blocks = {}
            for i, block_id in enumerate(block_ids):
                block = self.blocks[block_id]
                oriented_blocks[block_id] = block.with_orientation(orientations[i])
            
            # Store original blocks and use oriented ones
            original_blocks = self.blocks
            self.blocks = oriented_blocks
            
            # First, place any blocks with absolute constraints
            self.placed_blocks = {}
            self.occupied_positions = set()
            
            abs_constraint_map = {ac.block_id: ac.position for ac in self.absolute_constraints}
            blocks_with_abs_constraints = set(abs_constraint_map.keys())
            
            # Place absolutely constrained blocks first
            valid_abs_placement = True
            for block_id, position in abs_constraint_map.items():
                if self._is_valid_placement(block_id, position):
                    self.placed_blocks[block_id] = position
                    self.occupied_positions.update(
                        self.blocks[block_id].get_occupied_positions(position)
                    )
                else:
                    valid_abs_placement = False
                    break
            
            if not valid_abs_placement:
                self.blocks = original_blocks
                continue
            
            # Get remaining blocks to place
            remaining_initial = [b for b in block_ids if b not in blocks_with_abs_constraints]
            
            # If all blocks have absolute constraints, just verify relative constraints
            if not remaining_initial:
                if self._verify_all_constraints():
                    self.blocks = original_blocks
                    self.constraints = original_constraints
                    return True, self._create_placement_result()
                else:
                    self.blocks = original_blocks
                    continue
            
            # Try different starting positions for remaining blocks
            for start_x in range(0, min(10, self.grid_width)):
                for start_y in range(0, min(10, self.grid_depth)):
                    # Try different orderings of remaining blocks
                    for first_block_id in remaining_initial:
                        # Reset to just absolutely placed blocks
                        self.placed_blocks = dict(abs_constraint_map)
                        self.occupied_positions = set()
                        for block_id, pos in self.placed_blocks.items():
                            self.occupied_positions.update(
                                self.blocks[block_id].get_occupied_positions(pos)
                            )
                        
                        # Place first unconstrained block at starting position
                        start_pos = Position(start_x, start_y, 0)
                        if self._is_valid_placement(first_block_id, start_pos):
                            self.placed_blocks[first_block_id] = start_pos
                            self.occupied_positions.update(
                                self.blocks[first_block_id].get_occupied_positions(start_pos)
                            )
                        else:
                            continue
                        
                        # Create a queue of blocks to place
                        remaining_blocks = [b for b in remaining_initial if b != first_block_id]
                        placed_count = len(self.placed_blocks)
                        
                        # Keep trying to place blocks until no more can be placed
                        while remaining_blocks and placed_count < len(block_ids):
                            placed_in_round = False
                            
                            # First, try to place blocks connected to already-placed blocks
                            for block_id in remaining_blocks[:]:
                                # Find constraints involving this block
                                relevant_constraints = [c for c in self.constraints 
                                                       if c.block1_id == block_id or c.block2_id == block_id]
                                
                                # Try to place based on already placed blocks
                                for constraint in relevant_constraints:
                                    if constraint.block1_id in self.placed_blocks and constraint.block2_id == block_id:
                                        # block1 is placed, we need to place block2
                                        ref_pos = self.placed_blocks[constraint.block1_id]
                                        new_pos = self._calculate_adjacent_position(
                                            constraint.block1_id, ref_pos, 
                                            constraint.relation, 
                                            block_id
                                        )
                                        if new_pos and self._is_valid_placement(block_id, new_pos):
                                            self.placed_blocks[block_id] = new_pos
                                            self.occupied_positions.update(
                                                self.blocks[block_id].get_occupied_positions(new_pos)
                                            )
                                            remaining_blocks.remove(block_id)
                                            placed_count += 1
                                            placed_in_round = True
                                            break
                                    elif constraint.block2_id in self.placed_blocks and constraint.block1_id == block_id:
                                        # block2 is placed, we need to place block1
                                        # For "block1 relation block2", we place block1 in that relation to block2
                                        ref_pos = self.placed_blocks[constraint.block2_id]
                                        new_pos = self._calculate_position_for_constraint(
                                            block_id, constraint.relation, constraint.block2_id, ref_pos
                                        )
                                        if new_pos and self._is_valid_placement(block_id, new_pos):
                                            self.placed_blocks[block_id] = new_pos
                                            self.occupied_positions.update(
                                                self.blocks[block_id].get_occupied_positions(new_pos)
                                            )
                                            remaining_blocks.remove(block_id)
                                            placed_count += 1
                                            placed_in_round = True
                                            break
                            
                            # If no blocks were placed through constraints, try placing disconnected constraint groups
                            if not placed_in_round and remaining_blocks:
                                # Find if there are remaining blocks that form separate constraint groups
                                disconnected_groups = self._find_disconnected_constraint_groups(remaining_blocks)
                                
                                for group in disconnected_groups:
                                    if group:  # If group is not empty
                                        # Analyze constraints within this group to choose a good seed position
                                        seed_block, seed_position = self._calculate_smart_seed_position(group)
                                        
                                        if seed_block is not None and self._is_valid_placement(seed_block, seed_position):
                                            self.placed_blocks[seed_block] = seed_position
                                            self.occupied_positions.update(
                                                self.blocks[seed_block].get_occupied_positions(seed_position)
                                            )
                                            remaining_blocks.remove(seed_block)
                                            placed_count += 1
                                            placed_in_round = True
                                            break
                            
                            if not placed_in_round:
                                break
                        
                        # If all blocks placed and constraints satisfied, we found a solution
                        if placed_count == len(block_ids) and self._verify_all_constraints():
                            # Restore original blocks and constraints
                            self.blocks = original_blocks
                            if self.director_configs:
                                self.constraints = original_constraints
                            return True, self._create_placement_result()
            
            # Restore original blocks before trying next orientation
            self.blocks = original_blocks
        
        # Restore original constraints
        if self.director_configs:
            self.constraints = original_constraints
        print("Could not find a valid placement without negative positions")
        return False, None
    
    def _find_disconnected_constraint_groups(self, remaining_blocks: List[str]) -> List[List[str]]:
        """Find groups of blocks that are connected by constraints but disconnected from placed blocks"""
        if not remaining_blocks:
            return []
        
        # Build a graph of connections among remaining blocks
        graph = {}
        for block_id in remaining_blocks:
            graph[block_id] = set()
        
        # Add edges for constraints between remaining blocks
        for constraint in self.constraints:
            if constraint.block1_id in remaining_blocks and constraint.block2_id in remaining_blocks:
                graph[constraint.block1_id].add(constraint.block2_id)
                graph[constraint.block2_id].add(constraint.block1_id)
        
        # Find connected components using DFS
        visited = set()
        groups = []
        
        def dfs(node, current_group):
            if node in visited:
                return
            visited.add(node)
            current_group.append(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, current_group)
        
        for block_id in remaining_blocks:
            if block_id not in visited:
                group = []
                dfs(block_id, group)
                if group:
                    groups.append(group)
        
        return groups
    
    def _calculate_smart_seed_position(self, group: List[str]) -> Tuple[Optional[str], Position]:
        """Calculate a smart seed position for a disconnected constraint group.
        Analyzes constraints within the group to ensure there's room for all blocks."""
        
        if not group:
            return None, Position(0, 0, 0)
        
        # Find constraints within this group
        group_constraints = []
        for constraint in self.constraints:
            if constraint.block1_id in group and constraint.block2_id in group:
                group_constraints.append(constraint)
        
        # If no internal constraints, find a good position based on what's already placed
        if not group_constraints:
            # Find a position that doesn't conflict with already placed blocks
            # Start from (0,0,0) and search for a free spot
            for x in range(0, self.grid_width, 2):  # Step by 2 for block width
                for y in range(0, self.grid_depth, 2):
                    test_pos = Position(x, y, 0)
                    if self._is_valid_placement(group[0], test_pos):
                        return group[0], test_pos
            # Fallback - try with smaller steps if no position found
            for x in range(0, self.grid_width):
                for y in range(0, self.grid_depth):
                    test_pos = Position(x, y, 0)
                    if self._is_valid_placement(group[0], test_pos):
                        return group[0], test_pos
            # Last resort - return None to indicate failure
            return None, Position(0, 0, 0)
        
        # Analyze the constraint network to find a good seed block and position
        # Count how many blocks need to be placed in each direction from each block
        direction_requirements = {}
        for block_id in group:
            direction_requirements[block_id] = {
                'left_needed': 0, 'right_needed': 0,
                'front_needed': 0, 'back_needed': 0,
                'above_needed': 0, 'below_needed': 0
            }
        
        # Analyze each constraint to see what space is needed
        for constraint in group_constraints:
            block1_id = constraint.block1_id
            block2_id = constraint.block2_id
            relation = constraint.relation
            
            # For "block1 relation block2", block2 goes in the direction opposite to relation
            if relation == Direction.LEFT:
                # block1 LEFT block2, so block2 goes RIGHT of block1
                direction_requirements[block1_id]['right_needed'] += 1
                direction_requirements[block2_id]['left_needed'] += 1
            elif relation == Direction.RIGHT:
                # block1 RIGHT block2, so block2 goes LEFT of block1
                direction_requirements[block1_id]['left_needed'] += 1
                direction_requirements[block2_id]['right_needed'] += 1
            elif relation == Direction.FRONT:
                # block1 FRONT block2, so block2 goes BACK of block1
                direction_requirements[block1_id]['back_needed'] += 1
                direction_requirements[block2_id]['front_needed'] += 1
            elif relation == Direction.BACK:
                # block1 BACK block2, so block2 goes FRONT of block1
                direction_requirements[block1_id]['front_needed'] += 1
                direction_requirements[block2_id]['back_needed'] += 1
            elif relation == Direction.ABOVE:
                # block1 ABOVE block2, so block2 goes BELOW block1
                direction_requirements[block1_id]['below_needed'] += 1
                direction_requirements[block2_id]['above_needed'] += 1
            elif relation == Direction.BELOW:
                # block1 BELOW block2, so block2 goes ABOVE block1
                direction_requirements[block1_id]['above_needed'] += 1
                direction_requirements[block2_id]['below_needed'] += 1
        
        # Find the block that needs the least space in negative directions (good seed candidate)
        best_seed = None
        min_negative_requirements = float('inf')
        
        for block_id in group:
            # Calculate how much negative space this block needs
            negative_space_needed = (
                direction_requirements[block_id]['left_needed'] * 5 +  # Need space to the left
                direction_requirements[block_id]['front_needed'] * 5 + # Need space in front
                direction_requirements[block_id]['below_needed'] * 2   # Need space below
            )
            
            if negative_space_needed < min_negative_requirements:
                min_negative_requirements = negative_space_needed
                best_seed = block_id
        
        # If no good seed found, pick the first block
        if best_seed is None:
            best_seed = group[0]
        
        # Calculate seed position with enough margin
        reqs = direction_requirements[best_seed]
        margin_x = max(5, reqs['left_needed'] * 5)  # At least 5 units margin, more if needed
        margin_y = max(5, reqs['front_needed'] * 5)
        margin_z = max(0, reqs['below_needed'] * 2)
        
        # Try to find a valid position with these margins
        for test_x in range(margin_x, margin_x + 10):
            for test_y in range(margin_y, margin_y + 10):
                test_pos = Position(test_x, test_y, margin_z)
                if self._is_valid_placement(best_seed, test_pos):
                    return best_seed, test_pos
        
        # If no position found with margins, search more broadly within grid
        for x in range(0, self.grid_width, 2):
            for y in range(0, self.grid_depth, 2):
                test_pos = Position(x, y, margin_z)
                if self._is_valid_placement(best_seed, test_pos):
                    return best_seed, test_pos
        
        # Try all positions if still not found
        for x in range(0, self.grid_width):
            for y in range(0, self.grid_depth):
                for z in range(0, self.grid_height):
                    test_pos = Position(x, y, z)
                    if self._is_valid_placement(best_seed, test_pos):
                        return best_seed, test_pos
        
        # No valid position found
        return None, Position(0, 0, 0)
    
    def _calculate_position_for_constraint(self, block_to_place_id: str, relation: Direction, 
                                     ref_block_id: str, ref_pos: Position) -> Optional[Position]:
        """Calculate where to place block_to_place given 'block_to_place relation ref_block'
        
        Example: If constraint is "A Left B" and B is already placed:
        - block_to_place_id = "A", relation = LEFT, ref_block_id = "B"
        - This means A should be LEFT of B, so place A to the left of B
        """
        block_to_place = self.blocks[block_to_place_id]
        ref_block = self.blocks[ref_block_id]
        ref_w, ref_d, ref_h = ref_block.get_dimensions()
        new_w, new_d, new_h = block_to_place.get_dimensions()
        
        if self.strict_adjacent:
            # Strict mode: place blocks exactly touching
            if relation == Direction.LEFT:
                # block_to_place LEFT ref_block, so place block_to_place to the LEFT
                return Position(ref_pos.x - new_w, ref_pos.y, ref_pos.z)
            elif relation == Direction.RIGHT:
                # block_to_place RIGHT ref_block, so place block_to_place to the RIGHT
                return Position(ref_pos.x + ref_w, ref_pos.y, ref_pos.z)
            elif relation == Direction.FRONT:
                # block_to_place FRONT ref_block, so place block_to_place to the FRONT
                return Position(ref_pos.x, ref_pos.y - new_d, ref_pos.z)
            elif relation == Direction.BACK:
                # block_to_place BACK ref_block, so place block_to_place to the BACK
                return Position(ref_pos.x, ref_pos.y + ref_d, ref_pos.z)
            elif relation == Direction.BELOW:
                # block_to_place BELOW ref_block, so place block_to_place BELOW
                return Position(ref_pos.x, ref_pos.y, ref_pos.z - new_h)
            elif relation == Direction.ABOVE:
                # block_to_place ABOVE ref_block, so place block_to_place ABOVE
                return Position(ref_pos.x, ref_pos.y, ref_pos.z + ref_h)
        else:
            # Relaxed mode: find closest valid position that satisfies the constraint
            return self._find_closest_valid_position(block_to_place_id, ref_pos, relation, ref_block_id)
        
        return None

    def _find_closest_valid_position(self, new_block_id: str, ref_pos: Position, 
                                    relation: Direction, ref_block_id: str) -> Optional[Position]:
        """Find the closest valid position for new_block that satisfies the spatial relation
        relative to ref_block, considering existing placed blocks.
        
        IMPORTANT: relation here means where new_block should be placed relative to ref_block"""
        ref_block = self.blocks[ref_block_id]
        new_block = self.blocks[new_block_id]
        ref_w, ref_d, ref_h = ref_block.get_dimensions()
        new_w, new_d, new_h = new_block.get_dimensions()
        
        # Start with the touching position and search outward
        if relation == Direction.LEFT:
            # Place new_block to the LEFT of ref_block
            base_pos = Position(ref_pos.x - new_w, ref_pos.y, ref_pos.z)
            search_direction = (-1, 0, 0)  # Move further left
        elif relation == Direction.RIGHT:
            # Place new_block to the RIGHT of ref_block
            base_pos = Position(ref_pos.x + ref_w, ref_pos.y, ref_pos.z)
            search_direction = (1, 0, 0)   # Move further right
        elif relation == Direction.FRONT:
            # Place new_block to the FRONT of ref_block
            base_pos = Position(ref_pos.x, ref_pos.y - new_d, ref_pos.z)
            search_direction = (0, -1, 0)  # Move further front
        elif relation == Direction.BACK:
            # Place new_block to the BACK of ref_block
            base_pos = Position(ref_pos.x, ref_pos.y + ref_d, ref_pos.z)
            search_direction = (0, 1, 0)   # Move further back
        elif relation == Direction.BELOW:
            # Place new_block BELOW ref_block
            base_pos = Position(ref_pos.x, ref_pos.y, ref_pos.z - new_h)
            search_direction = (0, 0, -1)  # Move further down
        elif relation == Direction.ABOVE:
            # Place new_block ABOVE ref_block
            base_pos = Position(ref_pos.x, ref_pos.y, ref_pos.z + ref_h)
            search_direction = (0, 0, 1)   # Move further up
        else:
            return None
        
        # Try positions starting from touching and moving outward
        max_search_distance = 5  # Reasonable limit to avoid infinite search
        
        for distance in range(max_search_distance):
            candidate_pos = Position(
                base_pos.x + distance * search_direction[0],
                base_pos.y + distance * search_direction[1], 
                base_pos.z + distance * search_direction[2]
            )
            
            if self._is_valid_placement(new_block_id, candidate_pos):
                return candidate_pos
        
        return None  # No valid position found within search distance
    
    def _calculate_adjacent_position(self, ref_block_id: str, ref_pos: Position, 
                               relation: Direction, new_block_id: str) -> Optional[Position]:
        """Calculate position for new_block relative to ref_block based on the relation.
        
        IMPORTANT: This interprets the relation as the FINAL desired relationship.
        If relation is LEFT, it means ref_block should be LEFT of new_block,
        so new_block goes to the RIGHT of ref_block.
        
        Example: constraint "A Left B" with A already placed
        - ref_block_id = "A", relation = LEFT, new_block_id = "B"  
        - This should place B to the RIGHT of A (so A ends up left of B)
        """
        ref_block = self.blocks[ref_block_id]
        new_block = self.blocks[new_block_id]
        ref_w, ref_d, ref_h = ref_block.get_dimensions()
        new_w, new_d, new_h = new_block.get_dimensions()
        
        if self.strict_adjacent:
            # Strict mode: place blocks exactly touching
            if relation == Direction.LEFT:
                # ref_block LEFT new_block, so new_block goes RIGHT of ref_block
                return Position(ref_pos.x + ref_w, ref_pos.y, ref_pos.z)
            elif relation == Direction.RIGHT:
                # ref_block RIGHT new_block, so new_block goes LEFT of ref_block
                return Position(ref_pos.x - new_w, ref_pos.y, ref_pos.z)
            elif relation == Direction.FRONT:
                # ref_block FRONT new_block, so new_block goes BACK of ref_block
                return Position(ref_pos.x, ref_pos.y + ref_d, ref_pos.z)
            elif relation == Direction.BACK:
                # ref_block BACK new_block, so new_block goes FRONT of ref_block
                return Position(ref_pos.x, ref_pos.y - new_d, ref_pos.z)
            elif relation == Direction.BELOW:
                # ref_block BELOW new_block, so new_block goes ABOVE ref_block
                return Position(ref_pos.x, ref_pos.y, ref_pos.z + ref_h)
            elif relation == Direction.ABOVE:
                # ref_block ABOVE new_block, so new_block goes BELOW ref_block
                return Position(ref_pos.x, ref_pos.y, ref_pos.z - new_h)
        else:
            # Relaxed mode: find closest valid position that satisfies the constraint
            # Need to pass the inverse direction for finding valid position
            inverse_relation = self._reverse_direction(relation)
            return self._find_closest_valid_position(new_block_id, ref_pos, inverse_relation, ref_block_id)
        
        return None
    
    def _reverse_direction(self, direction: Direction) -> Direction:
        """Get the opposite direction"""
        opposites = {
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
            Direction.FRONT: Direction.BACK,
            Direction.BACK: Direction.FRONT,
            Direction.ABOVE: Direction.BELOW,
            Direction.BELOW: Direction.ABOVE
        }
        return opposites[direction]
    
    def _is_valid_placement(self, block_id: str, pos: Position) -> bool:
        """Check if placing block at position doesn't overlap with existing blocks and stays within grid"""
        # Check for negative positions
        if pos.x < 0 or pos.y < 0 or pos.z < 0:
            return False
            
        block = self.blocks[block_id]
        width, depth, height = block.get_dimensions()
        
        # Check if block would exceed grid boundaries
        if (pos.x + width > self.grid_width or 
            pos.y + depth > self.grid_depth or 
            pos.z + height > self.grid_height):
            return False
        
        new_positions = block.get_occupied_positions(pos)
        
        # Check if any of the new positions are already occupied
        overlap = new_positions.intersection(self.occupied_positions)
        if overlap:
            return False
        
        return True
    
    def _verify_all_constraints(self) -> bool:
        """Verify that all constraints are satisfied in current placement"""
        # First verify no overlaps
        all_occupied = set()
        for block_id, pos in self.placed_blocks.items():
            block_positions = self.blocks[block_id].get_occupied_positions(pos)
            overlap = all_occupied.intersection(block_positions)
            if overlap:
                print(f"Overlap found during verification: {block_id} overlaps at {list(overlap)[:4]}")
                return False
            all_occupied.update(block_positions)
        
        # Then verify constraints
        for constraint in self.constraints:
            if constraint.block1_id not in self.placed_blocks or \
               constraint.block2_id not in self.placed_blocks:
                return False
                
            pos1 = self.placed_blocks[constraint.block1_id]
            pos2 = self.placed_blocks[constraint.block2_id]
            
            # Check if blocks are adjacent
            if not self.blocks_are_adjacent(constraint.block1_id, pos1, 
                                           constraint.block2_id, pos2):
                return False
                
            # Check if relation is correct
            actual_relation = self.get_relative_direction(constraint.block1_id, pos1,
                                                         constraint.block2_id, pos2)
            if actual_relation != constraint.relation:
                return False
        
        return True
    
    def generate_build_instructions(self) -> List[str]:
        """Generate step-by-step build instructions"""
        if not self.placed_blocks:
            return ["No valid solution found"]
        
        instructions = []
        # Sort blocks by z-coordinate (build from bottom up)
        sorted_blocks = sorted(self.placed_blocks.items(), 
                             key=lambda x: (x[1].z, x[1].y, x[1].x))
        
        for block_id, pos in sorted_blocks:
            block = self.blocks[block_id]
            instructions.append(
                f"Place {block.color.value} {block.block_type.value} "
                f"({block.get_num_positions()} studs) "
                f"at position ({pos.x}, {pos.y}, {pos.z})"
            )
        
        return instructions
    
    def visualize_layer(self, z_level: int = 0) -> str:
        """Create a text visualization of a single layer"""
        if not self.placed_blocks:
            return "No blocks placed"
        
        # Find bounds across ALL layers for consistent spacing
        all_positions = set()
        for block_id, base_pos in self.placed_blocks.items():
            positions = self.blocks[block_id].get_occupied_positions(base_pos)
            all_positions.update(positions)
        
        if not all_positions:
            return "No blocks placed"
        
        # Get global bounds (across all layers)
        global_min_x = min(p.x for p in all_positions)
        global_max_x = max(p.x for p in all_positions)
        global_min_y = min(p.y for p in all_positions)
        global_max_y = max(p.y for p in all_positions)
        
        # Get positions for this specific layer
        layer_positions = set()
        for block_id, base_pos in self.placed_blocks.items():
            positions = self.blocks[block_id].get_occupied_positions(base_pos)
            layer_positions.update(p for p in positions if p.z == z_level)
        
        if not layer_positions:
            # Still create empty grid with same dimensions
            grid = []
            for y in range(global_min_y, global_max_y + 1):
                row = []
                for x in range(global_min_x, global_max_x + 1):
                    row.append('.')
                grid.append(' '.join(row))
            return '\n'.join(grid)
        
        # Create grid using global bounds
        grid = []
        for y in range(global_min_y, global_max_y + 1):
            row = []
            for x in range(global_min_x, global_max_x + 1):
                pos = Position(x, y, z_level)
                # Find which block occupies this position
                occupied_by = '.'
                for block_id, base_pos in self.placed_blocks.items():
                    if pos in self.blocks[block_id].get_occupied_positions(base_pos):
                        occupied_by = block_id[0]  # First letter of block ID
                        break
                row.append(occupied_by)
            grid.append(' '.join(row))
        
        return '\n'.join(grid)
    
    def _create_placement_result(self) -> Placement:
        """Create a Placement object with the current placement and 3D field"""
        if not self.placed_blocks:
            return Placement({}, [])
        
        # Find bounds
        all_positions = set()
        for block_id, base_pos in self.placed_blocks.items():
            positions = self.blocks[block_id].get_occupied_positions(base_pos)
            all_positions.update(positions)
        
        if not all_positions:
            return Placement(self.placed_blocks.copy(), [])
        
        # Get bounds
        min_x = min(p.x for p in all_positions)
        max_x = max(p.x for p in all_positions)
        min_y = min(p.y for p in all_positions)
        max_y = max(p.y for p in all_positions)
        min_z = min(p.z for p in all_positions)
        max_z = max(p.z for p in all_positions)
        
        # Create 3D field [z][y][x]
        field = []
        for z in range(min_z, max_z + 1):
            layer = []
            for y in range(min_y, max_y + 1):
                row = []
                for x in range(min_x, max_x + 1):
                    pos = Position(x, y, z)
                    # Find which block occupies this position
                    occupied_by = '.'
                    for block_id, base_pos in self.placed_blocks.items():
                        if pos in self.blocks[block_id].get_occupied_positions(base_pos):
                            occupied_by = block_id[0]  # First letter of block ID
                            break
                    row.append(occupied_by)
                layer.append(row)
            field.append(layer)
        
        return Placement(self.placed_blocks.copy(), field)