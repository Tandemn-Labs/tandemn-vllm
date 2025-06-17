import psutil
import pynvml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class GPUInfo:
    name: str
    total_vram_gb: float
    used_vram_gb: float
    free_vram_gb: float
    utilization_percent: Optional[float]
    temperature_celsius: Optional[float]

@dataclass
class SystemMetrics:
    cpu_percent: float
    ram_percent: float
    gpu_info: List[GPUInfo]
    timestamp: datetime

def bytes_to_gb(bytes_value: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes_value / (1024 ** 3)

def get_gpu_info() -> List[GPUInfo]:
    """
    Retrieve NVIDIA GPU information using NVML.
    Returns a list of GPUInfo objects with GPU details.
    """
    gpu_info_list = []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else raw_name
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                utilization = None
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temperature = None
            gpu_info_list.append(GPUInfo(
                name=name,
                total_vram_gb=round(bytes_to_gb(mem_info.total), 2),
                used_vram_gb=round(bytes_to_gb(mem_info.used), 2),
                free_vram_gb=round(bytes_to_gb(mem_info.free), 2),
                utilization_percent=utilization,
                temperature_celsius=temperature
            ))
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    return gpu_info_list

def get_system_metrics() -> SystemMetrics:
    """
    Get comprehensive system metrics including CPU, RAM, and GPU information.
    
    Returns:
        SystemMetrics object containing all system metrics
    """
    return SystemMetrics(
        cpu_percent=psutil.cpu_percent(interval=1),
        ram_percent=psutil.virtual_memory().percent,
        gpu_info=get_gpu_info(),
        timestamp=datetime.utcnow()
    )

def get_total_free_vram() -> float:
    """
    Calculate total free VRAM across all GPUs.
    
    Returns:
        Total free VRAM in GB
    """
    total_free = 0.0
    for gpu in get_gpu_info():
        total_free += gpu.free_vram_gb
    return round(total_free, 2)

def format_metrics_for_db(metrics: SystemMetrics) -> Dict[str, Any]:
    """
    Format system metrics for database storage.
    
    Args:
        metrics: SystemMetrics object
        
    Returns:
        Dictionary formatted for MongoDB storage
    """
    return {
        "cpu": metrics.cpu_percent,
        "ram": metrics.ram_percent,
        "gpu_info": [
            {
                "name": gpu.name,
                "total_vram_gb": gpu.total_vram_gb,
                "used_vram_gb": gpu.used_vram_gb,
                "free_vram_gb": gpu.free_vram_gb,
                "utilization_percent": gpu.utilization_percent,
                "temperature_celsius": gpu.temperature_celsius
            }
            for gpu in metrics.gpu_info
        ],
        "total_free_vram_gb": get_total_free_vram(),
        "timestamp": metrics.timestamp
    } 