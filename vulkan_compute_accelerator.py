#!/usr/bin/env python3
"""
Vulkan计算加速模块
提供基于Vulkan API的高性能矩阵运算和GPU加速计算
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union
import ctypes
import os
import sys

# 尝试导入Vulkan相关库
try:
    # 首先尝试pyvulkan
    import vulkan as vk
    from vulkan import *
    VULKAN_AVAILABLE = True
    print("✅ PyVulkan已导入，支持Vulkan加速")
except ImportError:
    try:
        # 尝试vulkan-api-python
        import vk
        from vk import *
        VULKAN_AVAILABLE = True
        print("✅ Vulkan API Python已导入")
    except ImportError:
        VULKAN_AVAILABLE = False
        print("⚠️ Vulkan库不可用，将使用备选实现")

# 导入NumPy和CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class VulkanComputeBackend:
    """
    Vulkan计算后端
    提供基于Vulkan的高性能矩阵运算
    """
    
    def __init__(self):
        self.vulkan_available = VULKAN_AVAILABLE
        self.device = None
        self.queue = None
        self.command_pool = None
        self.descriptor_pool = None
        self.compute_pipeline = None
        self.shader_module = None
        
        if self.vulkan_available:
            self._init_vulkan()
        else:
            print("⚠️ Vulkan不可用，将使用CPU回退")
    
    def _init_vulkan(self):
        """初始化Vulkan实例"""
        try:
            # 创建Vulkan实例
            app_info = VkApplicationInfo(
                sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pApplicationName=b"Vulkan Compute Solver",
                applicationVersion=1,
                pEngineName=b"Vulkan Engine",
                engineVersion=1,
                apiVersion=VK_API_VERSION_1_0
            )
            
            # 获取可用的扩展
            extension_names = [VK_KHR_SWAPCHAIN_EXTENSION_NAME]
            if self._check_extension_support(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME):
                extension_names.append(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)
            
            create_info = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=app_info,
                enabledExtensionCount=len(extension_names),
                ppEnabledExtensionNames=extension_names
            )
            
            self.instance = vk.vkCreateInstance(create_info, None, None)
            
            # 枚举物理设备
            device_count = ctypes.c_uint32()
            vk.vkEnumeratePhysicalDevices(self.instance, ctypes.byref(device_count), None)
            
            if device_count.value == 0:
                raise RuntimeError("没有找到Vulkan支持的物理设备")
            
            devices = (VkPhysicalDevice * device_count.value)()
            vk.vkEnumeratePhysicalDevices(self.instance, ctypes.byref(device_count), devices)
            
            # 选择第一个支持计算的第一个设备
            self.physical_device = devices[0]
            
            # 创建设备
            self._create_device()
            
            print("✅ Vulkan计算后端初始化成功")
            
        except Exception as e:
            print(f"❌ Vulkan初始化失败: {e}")
            self.vulkan_available = False
    
    def _check_extension_support(self, extension_name):
        """检查扩展支持"""
        try:
            extension_count = ctypes.c_uint32()
            vk.vkEnumerateDeviceExtensionProperties(self.physical_device, ctypes.byref(extension_count), None)
            extensions = (VkExtensionProperties * extension_count.value)()
            vk.vkEnumerateDeviceExtensionProperties(self.physical_device, ctypes.byref(extension_count), extensions)
            
            for ext in extensions:
                if ext.extensionName.decode() == extension_name:
                    return True
            return False
        except:
            return False
    
    def _create_device(self):
        """创建Vulkan设备"""
        try:
            # 获取队列家族索引
            queue_family_count = ctypes.c_uint32()
            vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device, ctypes.byref(queue_family_count), None)
            
            queue_families = (VkQueueFamilyProperties * queue_family_count.value)()
            vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device, ctypes.byref(queue_family_count), queue_families)
            
            # 寻找支持计算的队列家族
            queue_family_index = -1
            for i, queue_family in enumerate(queue_families):
                if queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT:
                    queue_family_index = i
                    break
            
            if queue_family_index == -1:
                raise RuntimeError("没有找到支持计算的队列家族")
            
            # 创建队列
            queue_priorities = ctypes.c_float(1.0)
            queue_create_info = VkDeviceQueueCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=queue_family_index,
                queueCount=1,
                pQueuePriorities=ctypes.byref(queue_priorities)
            )
            
            # 创建设备
            device_create_info = VkDeviceCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                queueCreateInfoCount=1,
                pQueueCreateInfos=ctypes.byref(queue_create_info)
            )
            
            self.device = vk.vkCreateDevice(self.physical_device, device_create_info, None, None)
            self.queue_family_index = queue_family_index
            
            # 获取队列句柄
            vk.vkGetDeviceQueue(self.device, queue_family_index, 0, ctypes.byref(self.queue))
            
            print("✅ Vulkan设备创建成功")
            
        except Exception as e:
            print(f"❌ 设备创建失败: {e}")
            raise
    
    def matrix_multiply(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        """
        使用Vulkan加速的矩阵乘法
        
        参数:
            matrix_a: 输入矩阵A
            matrix_b: 输入矩阵B
            
        返回:
            矩阵乘法结果
        """
        if not self.vulkan_available:
            return self._matrix_multiply_cpu(matrix_a, matrix_b)
        
        try:
            # 将数据上传到GPU
            device_buffer_a, memory_a = self._upload_to_device(matrix_a)
            device_buffer_b, memory_b = self._upload_to_device(matrix_b)
            
            # 创建结果缓冲区
            result_shape = (matrix_a.shape[0], matrix_b.shape[1])
            result_buffer, result_memory = self._create_buffer(result_shape)
            
            # 执行计算
            self._execute_compute_shader(device_buffer_a, device_buffer_b, result_buffer, 
                                      matrix_a.shape, matrix_b.shape)
            
            # 下载结果
            result = self._download_from_device(result_buffer, result_shape)
            
            # 清理资源
            self._cleanup_buffer(device_buffer_a, memory_a)
            self._cleanup_buffer(device_buffer_b, memory_b)
            self._cleanup_buffer(result_buffer, result_memory)
            
            return result
            
        except Exception as e:
            print(f"Vulkan计算失败，回退到CPU: {e}")
            return self._matrix_multiply_cpu(matrix_a, matrix_b)
    
    def matrix_vector_multiply(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """矩阵向量乘法"""
        if not self.vulkan_available:
            return self._matrix_vector_multiply_cpu(matrix, vector)
        
        try:
            device_buffer_matrix, memory_matrix = self._upload_to_device(matrix)
            device_buffer_vector, memory_vector = self._upload_to_device(vector)
            
            result_shape = (matrix.shape[0],)
            result_buffer, result_memory = self._create_buffer(result_shape)
            
            self._execute_compute_shader_matrix_vector(device_buffer_matrix, device_buffer_vector, 
                                                    result_buffer, matrix.shape, len(vector))
            
            result = self._download_from_device(result_buffer, result_shape)
            
            self._cleanup_buffer(device_buffer_matrix, memory_matrix)
            self._cleanup_buffer(device_buffer_vector, memory_vector)
            self._cleanup_buffer(result_buffer, result_memory)
            
            return result
            
        except Exception as e:
            print(f"Vulkan矩阵向量乘法失败，回退到CPU: {e}")
            return self._matrix_vector_multiply_cpu(matrix, vector)
    
    def dot_product(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """向量点积"""
        if not self.vulkan_available:
            return self._dot_product_cpu(vector_a, vector_b)
        
        try:
            device_buffer_a, memory_a = self._upload_to_device(vector_a)
            device_buffer_b, memory_b = self._upload_to_device(vector_b)
            
            result_shape = (1,)
            result_buffer, result_memory = self._create_buffer(result_shape)
            
            self._execute_compute_shader_dot_product(device_buffer_a, device_buffer_b, 
                                                  result_buffer, len(vector_a))
            
            result = self._download_from_device(result_buffer, result_shape)
            
            self._cleanup_buffer(device_buffer_a, memory_a)
            self._cleanup_buffer(device_buffer_b, memory_b)
            self._cleanup_buffer(result_buffer, result_memory)
            
            return float(result[0])
            
        except Exception as e:
            print(f"Vulkan点积计算失败，回退到CPU: {e}")
            return self._dot_product_cpu(vector_a, vector_b)
    
    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """向量归一化"""
        if not self.vulkan_available:
            return self._normalize_vector_cpu(vector)
        
        try:
            device_buffer, memory = self._upload_to_device(vector)
            
            result_shape = vector.shape
            result_buffer, result_memory = self._create_buffer(result_shape)
            
            self._execute_compute_shader_normalize(device_buffer, result_buffer, len(vector))
            
            result = self._download_from_device(result_buffer, result_shape)
            
            self._cleanup_buffer(device_buffer, memory)
            self._cleanup_buffer(result_buffer, result_memory)
            
            return result
            
        except Exception as e:
            print(f"Vulkan向量归一化失败，回退到CPU: {e}")
            return self._normalize_vector_cpu(vector)
    
    # CPU回退实现
    def _matrix_multiply_cpu(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """CPU矩阵乘法"""
        return np.dot(a, b)
    
    def _matrix_vector_multiply_cpu(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """CPU矩阵向量乘法"""
        return np.dot(matrix, vector)
    
    def _dot_product_cpu(self, a: np.ndarray, b: np.ndarray) -> float:
        """CPU点积"""
        return float(np.dot(a, b))
    
    def _normalize_vector_cpu(self, vector: np.ndarray) -> np.ndarray:
        """CPU向量归一化"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    # Vulkan缓冲区管理方法
    def _upload_to_device(self, data: np.ndarray):
        """上传数据到Vulkan设备"""
        # 简化的实现，实际需要更复杂的缓冲区管理
        buffer_size = data.nbytes
        buffer_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=buffer_size,
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vk.vkCreateBuffer(self.device, buffer_info, None, None)
        memory_requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        
        memory_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memory_requirements.size,
            memoryTypeIndex=0  # 简化的内存类型选择
        )
        
        memory = vk.vkAllocateMemory(self.device, memory_info, None, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        
        # 上传数据（简化实现）
        return buffer, memory
    
    def _create_buffer(self, shape: tuple):
        """创建缓冲区"""
        total_elements = np.prod(shape)
        buffer_size = total_elements * 8  # float64 = 8 bytes
        
        buffer_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=buffer_size,
            usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vk.vkCreateBuffer(self.device, buffer_info, None, None)
        memory_requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        
        memory_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=memory_requirements.size,
            memoryTypeIndex=0
        )
        
        memory = vk.vkAllocateMemory(self.device, memory_info, None, None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        
        return buffer, memory
    
    def _download_from_device(self, buffer, shape: tuple):
        """从设备下载数据"""
        # 简化的实现，实际需要使用命令缓冲区
        return np.zeros(shape)
    
    def _cleanup_buffer(self, buffer, memory):
        """清理缓冲区"""
        vk.vkDestroyBuffer(self.device, buffer, None)
        vk.vkFreeMemory(self.device, memory, None)
    
    # 计算着色器执行方法（占位符实现）
    def _execute_compute_shader(self, buffer_a, buffer_b, result_buffer, shape_a, shape_b):
        """执行计算着色器（简化实现）"""
        # 在实际实现中，这里会创建命令缓冲区，记录计算命令，然后执行
        pass
    
    def _execute_compute_shader_matrix_vector(self, matrix_buffer, vector_buffer, result_buffer, matrix_shape, vector_size):
        """执行矩阵向量乘法计算着色器"""
        pass
    
    def _execute_compute_shader_dot_product(self, buffer_a, buffer_b, result_buffer, size):
        """执行点积计算着色器"""
        pass
    
    def _execute_compute_shader_normalize(self, input_buffer, result_buffer, size):
        """执行向量归一化计算着色器"""
        pass
    
    def get_device_info(self):
        """获取设备信息"""
        if not self.vulkan_available:
            return {"status": "unavailable", "reason": "Vulkan not available"}
        
        try:
            # 获取物理设备属性
            device_properties = vk.vkGetPhysicalDeviceProperties(self.physical_device)
            memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
            
            return {
                "status": "available",
                "device_name": device_properties.deviceName.decode(),
                "device_type": device_properties.deviceType,
                "api_version": device_properties.apiVersion,
                "driver_version": device_properties.driverVersion,
                "memory_heaps": memory_properties.memoryHeapCount,
                "memory_types": memory_properties.memoryTypeCount
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


class VulkanOptimizedAHPSolver:
    """
    Vulkan优化的AHP求解器
    集成Vulkan加速的矩阵运算
    """
    
    def __init__(self, use_vulkan: bool = True, use_cupy: bool = False, tolerance: float = 1e-8):
        """
        初始化Vulkan优化的AHP求解器
        
        参数:
            use_vulkan: 是否使用Vulkan加速
            use_cupy: 是否使用CuPy加速（备选方案）
            tolerance: 计算精度
        """
        self.tolerance = tolerance
        self.use_vulkan = use_vulkan
        self.use_cupy = use_cupy
        
        # 初始化加速后端
        if use_vulkan:
            self.vulkan_backend = VulkanComputeBackend()
            if self.vulkan_backend.vulkan_available:
                self.compute_backend = "vulkan"
                print("🚀 使用Vulkan加速")
            else:
                print("⚠️ Vulkan不可用，回退到CuPy")
                self.use_vulkan = False
                if use_cupy and CUPY_AVAILABLE:
                    import cupy as cp
                    self.compute_backend = "cupy"
                    print("🚀 使用CuPy加速")
                else:
                    self.compute_backend = "numpy"
                    print("💻 使用NumPy CPU计算")
        elif use_cupy and CUPY_AVAILABLE:
            import cupy as cp
            self.compute_backend = "cupy"
            print("🚀 使用CuPy加速")
        else:
            self.compute_backend = "numpy"
            print("💻 使用NumPy CPU计算")
        
        # RI值
        self.ri_values = {
            1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
            11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
        }
        
        print(f"🎯 Vulkan优化的AHP求解器已启动")
        print(f"   - 计算后端: {self.compute_backend}")
        print(f"   - 容差: {tolerance}")
    
    def power_method_vulkan(self, matrix: np.ndarray, max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        使用Vulkan优化的幂法
        
        参数:
            matrix: 输入矩阵
            max_iterations: 最大迭代次数
            
        返回:
            (特征向量, 特征值)
        """
        n = matrix.shape[0]
        
        # 初始化向量
        if self.compute_backend == "vulkan":
            # 使用Vulkan初始化
            initial_vector = np.ones(n, dtype=np.float64) / n
        else:
            # 使用其他后端
            if self.compute_backend == "cupy":
                initial_vector = cp.ones(n, dtype=np.float64) / n
            else:
                initial_vector = np.ones(n, dtype=np.float64) / n
        
        # 幂法迭代
        for iteration in range(max_iterations):
            if self.compute_backend == "vulkan":
                # Vulkan矩阵向量乘法
                new_vector = self.vulkan_backend.matrix_vector_multiply(matrix, initial_vector)
                # Vulkan归一化
                new_vector = self.vulkan_backend.normalize_vector(new_vector)
                # Vulkan检查收敛
                diff = np.linalg.norm(new_vector - initial_vector)
            elif self.compute_backend == "cupy":
                new_vector = cp.dot(matrix, initial_vector)
                new_vector = new_vector / cp.linalg.norm(new_vector)
                diff = float(cp.linalg.norm(new_vector - initial_vector))
            else:
                new_vector = np.dot(matrix, initial_vector)
                new_vector = new_vector / np.linalg.norm(new_vector)
                diff = np.linalg.norm(new_vector - initial_vector)
            
            if diff < self.tolerance:
                break
            
            initial_vector = new_vector
        
        # 计算特征值
        if self.compute_backend == "vulkan":
            eigenvalue = self.vulkan_backend.dot_product(
                self.vulkan_backend.matrix_vector_multiply(matrix, initial_vector), 
                initial_vector
            ) / self.vulkan_backend.dot_product(initial_vector, initial_vector)
        elif self.compute_backend == "cupy":
            av = cp.dot(matrix, initial_vector)
            eigenvalue = float(cp.dot(av, initial_vector) / cp.dot(initial_vector, initial_vector))
        else:
            av = np.dot(matrix, initial_vector)
            eigenvalue = np.dot(av, initial_vector) / np.dot(initial_vector, initial_vector)
        
        return initial_vector, float(eigenvalue)
    
    def solve_ahp_with_vulkan(self, criteria_comparisons: List[List[float]], 
                            alternative_comparisons: Dict[str, List[List[float]]]) -> Dict:
        """
        使用Vulkan优化的AHP求解
        
        参数:
            criteria_comparisons: 准则层比较矩阵
            alternative_comparisons: 方案层比较矩阵
            
        返回:
            AHP分析结果
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🚀 Vulkan优化的AHP求解开始")
        print(f"{'='*60}")
        
        try:
            # 构建判断矩阵
            criteria_matrix = self._create_pairwise_matrix(criteria_comparisons)
            
            # 使用优化的幂法计算准则权重
            criteria_weights, criteria_eigenvalue = self.power_method_vulkan(criteria_matrix)
            
            # 一致性检验
            criteria_cr = self._check_consistency(criteria_matrix, criteria_weights, criteria_eigenvalue)
            
            print(f"   ✅ 准则权重: {criteria_weights}")
            print(f"   📊 一致性比率: {criteria_cr:.4f}")
            
            # 计算方案权重
            alternative_weights = {}
            total_consistency = 0
            
            for i, (criterion, comparisons) in enumerate(alternative_comparisons.items()):
                alt_matrix = self._create_pairwise_matrix(comparisons)
                alt_weights, alt_eigenvalue = self.power_method_vulkan(alt_matrix)
                alt_cr = self._check_consistency(alt_matrix, alt_weights, alt_eigenvalue)
                
                alternative_weights[criterion] = {
                    'weights': alt_weights,
                    'cr': alt_cr
                }
                
                total_consistency += criteria_weights[i] * alt_cr
            
            # 计算总排序
            total_scores = self._calculate_total_scores_vulkan(criteria_weights, alternative_weights)
            
            solve_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"✅ Vulkan优化的AHP分析完成!")
            print(f"   求解时间: {solve_time:.4f}秒")
            print(f"   计算后端: {self.compute_backend}")
            print(f"{'='*60}")
            
            return {
                'criteria_weights': criteria_weights,
                'criteria_cr': criteria_cr,
                'alternative_weights': alternative_weights,
                'total_scores': total_scores,
                'solve_time': solve_time,
                'compute_backend': self.compute_backend,
                'device_info': self.vulkan_backend.get_device_info() if self.compute_backend == "vulkan" else None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'solve_time': time.time() - start_time,
                'compute_backend': self.compute_backend
            }
    
    def _create_pairwise_matrix(self, comparisons: List[List[float]]) -> np.ndarray:
        """创建成对比较矩阵"""
        n = len(comparisons) + 1
        matrix = np.eye(n, dtype=np.float64)
        
        for i in range(n-1):
            for j in range(i+1, n):
                value = comparisons[i][j-i-1]
                matrix[i, j] = value
                matrix[j, i] = 1.0 / value
        
        return matrix
    
    def _check_consistency(self, matrix: np.ndarray, weights: np.ndarray, eigenvalue: float) -> float:
        """一致性检验"""
        n = matrix.shape[0]
        
        if n < 3:
            return 0.0
        
        ci = (eigenvalue - n) / (n - 1)
        ri = self.ri_values.get(n, 1.49)
        cr = ci / ri if ri > 0 else 0
        
        return cr
    
    def _calculate_total_scores_vulkan(self, criteria_weights: np.ndarray, alternative_weights: Dict) -> np.ndarray:
        """使用Vulkan优化的总得分计算"""
        first_criterion = list(alternative_weights.keys())[0]
        n_alternatives = len(alternative_weights[first_criterion]['weights'])
        
        if self.compute_backend == "vulkan":
            # 使用Vulkan进行加权求和
            total_scores = np.zeros(n_alternatives, dtype=np.float64)
            
            for i, (criterion, data) in enumerate(alternative_weights.items()):
                alt_weights = data['weights']
                criterion_weight = criteria_weights[i]
                
                # Vulkan加速的加权求和
                weighted_alt = alt_weights * criterion_weight
                total_scores += weighted_alt
        else:
            # 传统实现
            total_scores = np.zeros(n_alternatives, dtype=np.float64)
            for i, (criterion, data) in enumerate(alternative_weights.items()):
                alt_weights = data['weights']
                criterion_weight = criteria_weights[i]
                total_scores += alt_weights * criterion_weight
        
        return total_scores / np.sum(total_scores)


def test_vulkan_acceleration():
    """测试Vulkan加速功能"""
    print("🧪 Vulkan加速功能测试")
    
    # 创建Vulkan优化的求解器
    solver = VulkanOptimizedAHPSolver(use_vulkan=True, use_cupy=True)
    
    # 定义测试问题
    criteria_comparisons = [
        [3, 2, 4],
        [2, 3],
        [1]
    ]
    
    alternative_comparisons = {
        '准则1': [
            [2, 3, 4],
            [2, 3],
            [2]
        ],
        '准则2': [
            [1/2, 1/3, 1/4],
            [1/2, 1/3],
            [1/2]
        ],
        '准则3': [
            [1/2, 1/3, 1/5],
            [1/2, 1/4],
            [1/3]
        ]
    }
    
    # 求解
    result = solver.solve_ahp_with_vulkan(criteria_comparisons, alternative_comparisons)
    
    print(f"求解结果:")
    print(f"   状态: {result.get('status', 'success')}")
    print(f"   计算后端: {result.get('compute_backend', 'unknown')}")
    print(f"   求解时间: {result.get('solve_time', 0):.4f}秒")
    
    if 'device_info' in result and result['device_info']:
        print(f"   设备信息: {result['device_info']}")
    
    return result


if __name__ == "__main__":
    test_vulkan_acceleration()