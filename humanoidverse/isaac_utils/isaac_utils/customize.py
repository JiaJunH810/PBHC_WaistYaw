import torch

def batch_local_to_world_com(body_local_com, rigid_body_pos, rigid_body_rot):
        """
        将局部质心批量转换为世界坐标系下的质心
        
        参数:
            body_local_com: 局部质心坐标，shape为[env_num, body_num, 3]
            rigid_body_pos: 刚体位置，shape为[env_num, body_num, 3]
            rigid_body_rot: 刚体旋转四元数(xyzw)，shape为[env_num, body_num, 4]
        
        返回:
            world_centers: 世界坐标系下的质心，shape为[env_num, body_num, 3]
        """
        # 获取批量维度
        env_num, body_num = rigid_body_pos.shape[0], rigid_body_pos.shape[1]
        
        # 1. 将四元数(xyzw)转换为旋转矩阵 [env, body, 3, 3]
        # 注意：这里假设使用PyTorch的quaternion_to_matrix，需确保四元数格式匹配
        # 如果没有现成函数，可自行实现批量四元数转旋转矩阵
        rotation_matrices = quaternion_to_matrix(rigid_body_rot)  # [env, body, 3, 3]
        
        # 2. 构造批量齐次变换矩阵 [env, body, 4, 4]
        # 初始化单位矩阵
        T = torch.eye(4, device=rigid_body_pos.device, dtype=rigid_body_pos.dtype)
        T = T.repeat(env_num, body_num, 1, 1)  # [env, body, 4, 4]
        
        # 填充旋转矩阵部分
        T[..., :3, :3] = rotation_matrices
        
        # 填充平移部分
        T[..., :3, 3] = rigid_body_pos  # [env, body, 3] 广播到 [env, body, 3]
        
        # 3. 局部质心转换为齐次坐标 [env, body, 4]
        local_com_homo = torch.cat([
            body_local_com, 
            torch.ones_like(body_local_com[..., :1])  # 增加w=1
        ], dim=-1)  # [env, body, 4]
        
        # 4. 转换为列向量 [env, body, 4, 1]
        local_com_homo = local_com_homo.unsqueeze(-1)
        
        # 5. 齐次变换：T @ 局部质心
        world_com_homo = torch.matmul(T, local_com_homo)  # [env, body, 4, 1]
        
        # 6. 提取前3个分量，恢复为3D坐标
        world_centers = world_com_homo[..., :3, 0]  # [env, body, 3]
        
        return world_centers

# 补充：四元数(xyzw)转旋转矩阵的批量实现

def quaternion_to_matrix(quaternions):
    """
    将批量四元数(xyzw)转换为旋转矩阵
    
    参数:
        quaternions: 四元数，shape为[..., 4]
    
    返回:
        旋转矩阵，shape为[..., 3, 3]
    """
    x, y, z, w = torch.unbind(quaternions, dim=-1)
    two_x = 2.0 * x
    two_y = 2.0 * y
    two_z = 2.0 * z
    two_xx = two_x * x
    two_xy = two_x * y
    two_xz = two_x * z
    two_xw = two_x * w
    two_yy = two_y * y
    two_yz = two_y * z
    two_yw = two_y * w
    two_zz = two_z * z
    two_zw = two_z * w

    # 构造旋转矩阵
    matrix = torch.stack([
        1.0 - two_yy - two_zz, two_xy - two_zw, two_xz + two_yw,
        two_xy + two_zw, 1.0 - two_xx - two_zz, two_yz - two_xw,
        two_xz - two_yw, two_yz + two_xw, 1.0 - two_xx - two_yy
    ], dim=-1).view(*quaternions.shape[:-1], 3, 3)
    
    return matrix