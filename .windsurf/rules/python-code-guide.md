---
trigger: glob
globs: *.py
---

请按照以下规范生成 Python 代码：

## 类型安全与 IDE 友好

1. **强类型要求**：
    - 所有 public 和 protected 方法的参数和返回值必须使用强类型注解
    - 禁止在 public/protected 方法签名中使用 `Any` 类型
    - 使用具体类型：`str`, `int`, `list[str]`, `dict[str, int]`, `Optional[T]`, `Union[T1, T2]` 等
    - 类属性也必须添加类型注解

2. **显式接口定义**：
    - 使用 Duck Typing 时，必须显式定义接口
    - 使用 `abc.ABC` 定义抽象基类，或使用 `typing.Protocol` 定义协议
    - 抽象方法使用 `@abstractmethod` 装饰器
    - 协议类使用 `@runtime_checkable` 装饰器（如需运行时检查）
    - 实现类必须显式继承抽象基类或协议

3. **IDE 提示优化**：
    - 确保所有类型注解完整，便于 IDE 自动补全和类型检查
    - 使用 `from __future__ import annotations` 支持前向引用
    - 导入类型时使用 `from typing import ...`

## 异常处理规范（重要）

1. **禁止滥用 try-except**：
    - 不要使用 try-except 作为正常的流程控制
    - 不要为了"保险"而到处加 try-except
    - 每个函数/方法的 try-except 数量应 ≤ 2 个
    - 如果需要更多，说明函数职责过重，应拆分

2. **明确异常类型**：
    - 禁止使用 `except Exception:` 或 `except:`（过于宽泛）
    - 必须捕获具体的异常类型，如：
        * `ValueError`: 参数值错误
        * `TypeError`: 类型错误
        * `KeyError`: 字典键不存在
        * `AttributeError`: 属性不存在
        * `IOError/OSError`: 文件/网络操作失败
        * `RuntimeError`: 运行时错误
    - 可以组合多个异常：`except (ValueError, TypeError) as e:`

3. **异常必须记录日志**：
    - 捕获异常后，必须使用 logger 记录，不得静默吞掉
    - 使用 `logger.error()` 或 `logger.warning()` 记录异常
    - 重要异常使用 `exc_info=True` 记录完整堆栈
    - 示例：
      ```python
      try:
          result = risky_operation()
      except (ValueError, TypeError) as e:
          logger.error(f"操作失败: {e}", exc_info=True)
          raise  # 或返回默认值
      ```

4. **异常处理策略**：
    - **预期异常**：捕获、记录、返回默认值或优雅降级
    - **意外异常**：记录后向上抛出（raise），让调用者处理
    - **不可恢复异常**：不要捕获（如 `KeyboardInterrupt`, `SystemExit`, `MemoryError`）

5. **优先使用 EAFP 而非 LBYL**：
    - EAFP (Easier to Ask for Forgiveness than Permission): 先尝试，失败再处理
    - LBYL (Look Before You Leap): 先检查，再执行
    - Python 推荐 EAFP，但要配合具体异常类型
    - 示例：
      ```python
      # ❌ LBYL（过度防御）
      if hasattr(obj, 'method'):
          try:
              obj.method()
          except Exception:
              pass
      
      # ✅ EAFP（Python 风格）
      try:
          obj.method()
      except AttributeError as e:
          logger.debug(f"对象不支持该方法: {e}")
      ```

6. **减少嵌套，提取辅助函数**：
    - 如果一个函数有多个 try-except，考虑拆分为多个小函数
    - 每个小函数处理一个具体的异常场景
    - 主函数负责协调调用，保持扁平结构

## 日志记录规范

1. **模块级 Logger**：
    - 每个模块在文件顶部（import 之后）定义 logger：
      ```python
      import logging
      
      logger = logging.getLogger(__name__)
      ```

2. **Logger 使用原则**：
    - 禁止在方法参数中传递 logger 对象
    - 每个类、每个方法直接使用模块级的 `logger`
    - 便于在任何位置快速添加日志，无需修改方法签名

3. **日志级别使用**：
    - DEBUG: 详细的调试信息（如参数值、中间状态）
    - INFO: 关键流程节点（如方法入口、重要操作完成）
    - WARNING: 警告信息（如使用了降级方案、非预期但可处理的情况）
    - ERROR: 错误信息（必须带异常堆栈 `exc_info=True`）

4. **日志记录位置**：
    - 公共方法的入口：记录关键参数（INFO 或 DEBUG）
    - 重要操作完成：记录结果（INFO）
    - 异常捕获处：记录异常详情（ERROR 或 WARNING）
    - 分支决策点：记录选择的路径（DEBUG）

## Python 之禅原则

- 优美胜于丑陋（Beautiful is better than ugly）
- 明确胜于隐晦（Explicit is better than implicit）
- 简单胜于复杂（Simple is better than complex）
- 扁平胜于嵌套（Flat is better than nested）
- 可读性很重要（Readability counts）
- 错误不应被静默（Errors should never pass silently）
- 除非明确要求静默（Unless explicitly silenced）

## 代码结构要求

1. **函数职责单一**：每个函数只做一件事
2. **嵌套层级 ≤ 3 层**：超过则拆分函数
3. **函数长度 ≤ 50 行**：超过则拆分
4. **使用提前返回**：避免深层嵌套的 if-else

## 其他规范

- 遵循 PEP 8 命名规范
- 使用 docstrings（Google 或 NumPy 格式）说明类和方法功能
- 使用上下文管理器处理资源
- 使用 `dataclasses` 或 `pydantic` 表示数据模型

## 示例代码结构

```python
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Protocol, Optional, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DataProcessor(Protocol):
    """数据处理器协议"""
    
    def process(self, data: str) -> dict[str, any]:
        """处理数据"""
        ...


class BaseService(ABC):
    """服务基类"""
    
    def __init__(self, name: str, timeout: int = 30) -> None:
        self.name: str = name
        self.timeout: int = timeout
        logger.info(f"初始化服务: {name}, 超时: {timeout}s")
    
    @abstractmethod
    def execute(self, request: dict[str, str]) -> dict[str, any]:
        """执行服务请求"""
        pass
    
    def _validate(self, data: dict[str, str]) -> bool:
        """验证数据（protected 方法也需要强类型）"""
        logger.debug(f"验证数据: {data}")
        return bool(data)


class ConcreteService(BaseService):
    """具体服务实现"""
    
    def __init__(self, name: str, processor: DataProcessor) -> None:
        super().__init__(name)
        self._processor: DataProcessor = processor
    
    def execute(self, request: dict[str, str]) -> dict[str, any]:
        """执行服务请求
        
        Args:
            request: 请求数据字典
            
        Returns:
            处理结果字典
            
        Raises:
            ValueError: 当请求数据无效时
        """
        logger.info(f"开始执行请求: {request.get('id', 'unknown')}")
        
        try:
            if not self._validate(request):
                raise ValueError("无效的请求数据")
            
            result = self._processor.process(request.get("data", ""))
            logger.info(f"请求执行成功")
            return result
            
        except Exception as e:
            logger.error(f"请求执行失败: {e}", exc_info=True)
            raise
```