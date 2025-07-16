"""
Base Service Class with Circuit Breaker Pattern
Provides foundation for all service implementations
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ServiceResult:
    """Standard result format for all services"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    service_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker pattern"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception(
                        f"Circuit breaker is OPEN for {func.__name__}"
                    )
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > 
            timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


class BaseService(ABC):
    """Base class for all services with common functionality"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.status = ServiceStatus.HEALTHY
        self.circuit_breaker = CircuitBreaker()
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "last_error": None,
            "last_success": None
        }
    
    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> ServiceResult:
        """Main processing method to be implemented by subclasses"""
        pass
    
    async def execute(self, request: Dict[str, Any]) -> ServiceResult:
        """Execute service with monitoring and error handling"""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Add circuit breaker protection
            result = await self._execute_with_circuit_breaker(request)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_success_metrics(processing_time)
            
            # Add metadata
            result.processing_time = processing_time
            result.service_name = self.service_name
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_failure_metrics(str(e))
            
            logger.error(f"{self.service_name} failed: {str(e)}")
            
            return ServiceResult(
                success=False,
                error=str(e),
                processing_time=processing_time,
                service_name=self.service_name,
                metadata={"request": request}
            )
    
    @CircuitBreaker()
    async def _execute_with_circuit_breaker(
        self, request: Dict[str, Any]
    ) -> ServiceResult:
        """Execute with circuit breaker protection"""
        return await self.process(request)
    
    def _update_success_metrics(self, processing_time: float):
        """Update metrics for successful request"""
        self.metrics["successful_requests"] += 1
        self.metrics["last_success"] = datetime.now()
        
        # Update rolling average
        total = self.metrics["successful_requests"]
        current_avg = self.metrics["average_processing_time"]
        self.metrics["average_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def _update_failure_metrics(self, error: str):
        """Update metrics for failed request"""
        self.metrics["failed_requests"] += 1
        self.metrics["last_error"] = {
            "error": error,
            "timestamp": datetime.now()
        }
        
        # Update service status based on failure rate
        total = self.metrics["total_requests"]
        failure_rate = (
            self.metrics["failed_requests"] / total if total > 0 else 0
        )
        
        if failure_rate > 0.5:
            self.status = ServiceStatus.UNHEALTHY
        elif failure_rate > 0.2:
            self.status = ServiceStatus.DEGRADED
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "service": self.service_name,
            "status": self.status.value,
            "circuit_state": self.circuit_breaker.state.value,
            "metrics": self.metrics
        }
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Subclasses can override for specific health checks
            return self.status != ServiceStatus.UNHEALTHY
        except Exception:
            return False 